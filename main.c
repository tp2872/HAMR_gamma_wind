/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal1, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/
#include "decs_MPI.h"
#include "defs.h"
//#include "cudaProfiler.h"


/*****************************************************************/
/*****************************************************************
   main():
   ------

     -- Initializes, time-steps, and concludes the simulation. 
     -- Handles timing of output routines;
     -- Main is main, what more can you say.  

-*****************************************************************/
int main(int argc, char *argv[])
{
	double tdump, tlog;
	int nfailed = 0;
	int i, j, z, u, n, l;
	double r, th, phi, X[NDIM];
	clock_t begin2;
	nstep = 0;
	defcon = 1.;

	/* Perform Initializations, either directly or via checkpoint */
	MPI_initialize(argc, argv);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_init();
	#endif
	set_AMR();

	if (!restart_read()) {
		#if(DEREFINE_POLE)
		derefine_pole();
		#endif
		for (l = 0; l < N_LEVELS_3D; l++) {
			init();
			#if(N_LEVELS_1D_INT>0 && D3>0)
			average_grid();
			#endif
			#if(N_LEVELS_3D>1)
			check_refcrit();
			#endif
		}	
	}

	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	#endif
	
	/* do initial diagnostics */
	#if(GPU_ENABLED || GPU_DEBUG )
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	#endif
	first_dump = 0;
	diag(LOG_OUT);

	DTl = 50.0;
	DTd = 50.0;
	tdump = t + DTd;
	tlog = t + DTl;
	tref = t;
	time_spent3 = 0.0;
	begin1 = get_wall_time();
	begin2 = begin1;

	//cuProfilerStart();
	while(t < tf) {
		/*Used for running OpenCL on either GPU or CPU*/
		#if(GPU_ENABLED && !GPU_DEBUG)
		GPU_step_ch();
		#endif
		#if(CPU_OPENMP)
		step_ch();
		#endif

		/*Used for debugging*/
		#if(GPU_DEBUG)
		step_ch_debug();
		#endif

		/* deal with failed timestep, exit upon failure */
		if (failed) break;

		//Every swithchtime read out data from GPU and set boundary
		if (nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0){
			end1 = get_wall_time();
			#if (GPU_ENABLED==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			bound_prim(p, 1);
			if (dt > 0.5) break;
		}

		//Refine every TREF
		if (t >= tref && nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0) {
			#if(N_LEVELS_3D>1)
			set_timelevel(1);
			check_refcrit();
			if (rank == 0) fprintf(stderr, "Refinement succesfull! \n");
			#else
			close_dump();
			close_rdump();
			close_gdump();
			set_timelevel(1);
			balance_load();
			#if(GPU_ENABLED)
			balance_load_gpu();
			#endif
			#endif
			tref += TREF;
		}

		//Put out log file and rdump file
		if (t >= tlog && nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0) {
			restart_write(); //do restart dump simultaneous with log
			tlog += DTl;
		}

		/* Put out dump file*/
		if (t >= tdump && nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0) {
			diag(DUMP_OUT) ;
			tdump += DTd;
		}	

		#if TIMER
		if (nstep % (DUMPFACTOR*AMR_SWITCHTIMELEVEL) == 0){
			diag(LOG_OUT);
			MPI_Allreduce(MPI_IN_PLACE, &ndt1, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt2, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt3, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			if (rank == 0){
				fprintf(stderr, "Runtime: %f MPI-time: %f ", (double)(end1 - begin1), time_spent3);
				fprintf(stderr, "dt1: %f dt2: %f dt3: %f nstep: %d \n", ndt1, ndt2, ndt3, nstep);
				fflush(stderr);
			}
			time_spent3 = 0.0;	

			//Safe and exit at end of 24 hour runtime
			if ((double)(begin2 - end1) > RUNTIME*3600.){
				if(rank==0) fprintf(stderr, "Finishing simulation after 24 hour time period! \n");
				restart_write();
				break;
			}
			begin1 = get_wall_time();
		}
		#endif
	}
	//cuProfilerStop();

	/* do final diagnostics */
	diag(FINAL_OUT) ;

	/*Close GPU*/
	for (n = 0; n < n_active; n++){
		free_arrays(n_ord[n]);
		GPU_finish(n_ord[n], 1);
	}
	return(0) ;
}

/*This function initialises the MPI structure. It divides the grid(N1, N2, N3) among the MPI processes.
The host node is node 0 by default.*/
void MPI_initialize(int argc, char *argv[])
{
	#if (MPI_enable)
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int i, j, z, len, dim, corn, rankloop;
	int dims[3], periods[3], coords[3];
	int rdma_direct = 0, local_rank = 0, threadid;

	/*Get basic initialisation*/
	rdma_direct = getenv("MPICH_RDMA_ENABLED_CUDA") == NULL ? 0 : atoi(getenv("MPICH_RDMA_ENABLED_CUDA"));
	if (getenv("MV2_COMM_WORLD_LOCAL_RANK") != NULL){
		local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK") == NULL ? 0 : atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
	}
	if (getenv("OMPI_COMM_WORLD_LOCAL_RANK") != NULL){
		local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK") == NULL ? 0 : atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
	}
	#if(GPU_ENABLED)
	cudaGetDeviceCount(&numdevices);
	cudaSetDevice(local_rank%numdevices);
	#endif
	rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &i);

	if (rc != MPI_SUCCESS) {
		fprintf(stderr, "Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Get_processor_name(hostname, &len);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	mpi_cartcomm = MPI_COMM_WORLD;
	MPI_Comm_split(mpi_cartcomm, rank, rank, &mpi_self);

	/*Give basic diagnostics*/
	if (rank == 0){
		if (rdma_direct != 1 && GPU_DIRECT == 1){
			fprintf(stderr, "MPICH_RDMA_ENABLED_CUDA not enabled but GPU_DIRECT still turned on!\n");
		}
		fprintf(stderr, "Number of MPI tasks: %d \nRunning on: %s\n", numtasks, hostname);
	}
	#endif

	#pragma omp parallel shared(nthreads) private(threadid)
	{
		threadid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		if (threadid == 0 && rank == 0) {
			fprintf(stderr, "nthreads = %d\n", nthreads);
		}
	}
	//omp_set_num_threads(1);

	if (rank == 0){
		system("mkdir dumps gdumps rdumps0 rdumps1");
	}
}

void mpi_synch(void){
	int i,n;
	//MPI_Barrier(MPI_COMM_WORLD);
	for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
			if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
			break;
		}
	}
}


/*****************************************************************/
/*****************************************************************
  set_arrays():
  ----------

       -- sets to zero all arrays, plus performs pointer trick 
          so that grid arrays can legitimately refer to ghost 
          zone quantities at positions  i = -2, -1, N1, N1+1 and 
          j = -2, -1, N2, N2+1

 *****************************************************************/
void set_arrays_image(void)
{
	
}

void set_arrays(int n)
{
	int i=0;

	//Find location in memory for new block and set nl[n]
	while (i < NB_LOCAL){
		if (mem_spot[i] != 1) break;
		i++;
		if (i == NB_LOCAL){
			fprintf(stderr, "Node %d ran out of local node memory! Stopping \n", rank);
			exit(0);
		}
	}
	nl[n] = i;

	if (mem_spot[i] == 0){
		alloc_bounds_CPU(n);
		mem_spot[i] = 1;
		return;
	}
	else mem_spot[i] = 1;

	array[nl[n]] = (float *)malloc(9 * BS_1*BS_2*BS_3 * sizeof(float));
	array_gdump1[nl[n]] = (double *)malloc(9 * BS_1*BS_2*BS_3 * sizeof(double));
	array_gdump2[nl[n]] = (double *)malloc(49 * BS_1*BS_2 * sizeof(double));
	array_rdump[nl[n]] = (double *)malloc((NPR + NDIM) * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	array_diag[nl[n]] = (float *)malloc(4 * BS_1*BS_2*BS_3 * sizeof(float));
	Katm[nl[n]] = (double(*))malloc((BS_1 + 2 * N1G) * sizeof(double));
	p[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	ph[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#if(STAGGERED)
	ps[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	psh[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	#if(LEER)
	V[nl[n]] = (double(*)[6])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double[6]));
	#endif
	dq[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F1[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F2[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F3[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	pflag[nl[n]] = (int(*))malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int));
	#if(CPU_OPENMP || 1)
	#if(STAGGERED)
	dE[nl[n]] = (double(*)[2][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[2][NDIM][NDIM]));
	#endif
	E_corn[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	failimage[nl[n]] = (int(*)[NFAIL])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int[NFAIL]));
	#if(!NSY)
	conn[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NDIM][NDIM][NDIM]));
	gcov[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	gcon[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	gdet[nl[n]] = (double(*)[NPG])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG]));
	#else
	conn[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	gcov[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NPG][NDIM][NDIM]));
	gcon[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG][NDIM][NDIM]));
	gdet[nl[n]] = (double(*)[NPG])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG]));
	#endif
	#if(HLLC)
	Mud[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	Mud_inv[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	#endif
	#if(ZIRI_DUMP)
	dump_buffer[nl[n]] = (double(*))malloc(BS_1 * BS_2 * BS_3 * 13 *sizeof(double));
	dxdxp_z[nl[n]] = (double(*)[NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	dxpdx_z[nl[n]] = (double(*)[NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	#endif
	#if (ELLIPTICAL2)
	dU_s[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPR]));
	#endif
	alloc_bounds_CPU(n);
}

void alloc_bounds_CPU(int n){
	int ref1_1, ref1_3, ref1_5, ref1_6;
	int ref2_2, ref2_4, ref2_5, ref2_6;
	int ref3_1, ref3_2, ref3_3, ref3_4;
	int ref1_1s, ref1_3s;
	int ref3_1s, ref3_3s, ref3_2s, ref3_4s;

	ref1_1 = REF_1; ref1_3 = REF_1; ref1_5 = REF_1; ref1_6 = REF_1;
	ref2_2 = REF_2; ref2_4 = REF_2; ref2_5 = REF_2; ref2_6 = REF_2;
	ref3_1 = REF_3; ref3_2 = REF_3; ref3_3 = REF_3; ref3_4 = REF_3;
	ref1_1s = REF_1; ref1_3s = REF_1;
	ref3_1s = REF_3; ref3_3s = REF_3;

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
	if ((block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*(int)pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) && DEREFINE_POLE){
		ref3_2s = 0;
		ref3_4s = 0;
	}
	
	send1[nl[n]] = (double *)malloc(NG *(1 + ref1_1)*(1 + ref3_1)* (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	send2[nl[n]] = (double *)malloc(NG *(1 + ref2_2)*(1 + ref3_2)* (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	send3[nl[n]] = (double *)malloc(NG *(1 + ref1_3)*(1 + ref3_3)* (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	send4[nl[n]] = (double *)malloc(NG *(1 + ref2_4)*(1 + ref3_4)* (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5[nl[n]] = (double *)malloc(NG *(1 + ref2_5)*(1 + ref1_5)* (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G) * (BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	send6[nl[n]] = (double *)malloc(NG *(1 + ref2_6)*(1 + ref1_6)* (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G) * (BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	#endif

	#if(N_LEVELS>1)
	send1_3[nl[n]] = send1[nl[n]];
	send1_4[nl[n]] = send1[nl[n]] + ref3_1*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send1_7[nl[n]] = send1[nl[n]] + (ref3_1 + ref1_1)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send1_8[nl[n]] = send1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send2_1[nl[n]] = send2[nl[n]];
	send2_2[nl[n]] = send2[nl[n]] + ref3_2*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send2_3[nl[n]] = send2[nl[n]] + (ref3_2 + ref2_2)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send2_4[nl[n]] = send2[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send3_1[nl[n]] = send3[nl[n]];
	send3_2[nl[n]] = send3[nl[n]] + ref3_3*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send3_5[nl[n]] = send3[nl[n]] + (ref3_3 + ref1_3)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send3_6[nl[n]] = send3[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send4_5[nl[n]] = send4[nl[n]];
	send4_6[nl[n]] = send4[nl[n]] + ref3_4*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	send4_7[nl[n]] = send4[nl[n]] + (ref3_4 + ref2_4)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	send4_8[nl[n]] = send4[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	#if(N3G>0)	
	send5_1[nl[n]] = send5[nl[n]];
	send5_3[nl[n]] = send5[nl[n]] + ref2_5*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send5_5[nl[n]] = send5[nl[n]] + (ref1_5 + ref2_5)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send5_7[nl[n]] = send5[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send6_2[nl[n]] = send6[nl[n]];
	send6_4[nl[n]] = send6[nl[n]] + ref2_6*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	send6_6[nl[n]] = send6[nl[n]] + (ref1_6 + ref2_6)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	send6_8[nl[n]] = send6[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	#endif
	#endif
	receive1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	receive6[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_4[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_7[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_8[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive2_1[nl[n]] = receive2[nl[n]];
	receive2_2[nl[n]] = receive2[nl[n]] + ref3_4*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_3[nl[n]] = receive2[nl[n]] + (ref3_4 + ref2_4)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_4[nl[n]] = receive2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive3_1[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_2[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_5[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_6[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive4_5[nl[n]] = receive4[nl[n]];
	receive4_6[nl[n]] = receive4[nl[n]] + ref3_2*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_7[nl[n]] = receive4[nl[n]] + (ref3_2 + ref2_2)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_8[nl[n]] = receive4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)	
	receive5_1[nl[n]] = receive5[nl[n]];
	receive5_3[nl[n]] = receive5[nl[n]] + ref2_6*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_5[nl[n]] = receive5[nl[n]] + (ref1_6 + ref2_6)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_7[nl[n]] = receive5[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive6_2[nl[n]] = receive6[nl[n]];
	receive6_4[nl[n]] = receive6[nl[n]] + ref2_5*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_6[nl[n]] = receive6[nl[n]] + (ref1_5 + ref2_5)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_8[nl[n]] = receive6[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#if(PRESTEP==-100 || PRESTEP2==-100)
	tempreceive1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* 2* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	tempreceive2[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	tempreceive3[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)*2* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	tempreceive4[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	tempreceive5[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	tempreceive6[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	tempreceive1_3[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_4[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_7[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_8[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive2_1[nl[n]] = tempreceive2[nl[n]];
	tempreceive2_2[nl[n]] = tempreceive2[nl[n]] + ref3_4*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive2_3[nl[n]] = tempreceive2[nl[n]] + (ref3_4 + ref2_4)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive2_4[nl[n]] = tempreceive2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive3_1[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_2[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_5[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_6[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive4_5[nl[n]] = tempreceive4[nl[n]];
	tempreceive4_6[nl[n]] = tempreceive4[nl[n]] + ref3_2*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempreceive4_7[nl[n]] = tempreceive4[nl[n]] + (ref3_2 + ref2_2)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempreceive4_8[nl[n]] = tempreceive4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)	
	tempreceive5_1[nl[n]] = tempreceive5[nl[n]];
	tempreceive5_3[nl[n]] = tempreceive5[nl[n]] + ref2_6*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive5_5[nl[n]] = tempreceive5[nl[n]] + (ref1_6 + ref2_6)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive5_7[nl[n]] = tempreceive5[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive6_2[nl[n]] = tempreceive6[nl[n]];
	tempreceive6_4[nl[n]] = tempreceive6[nl[n]] + ref2_5*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	tempreceive6_6[nl[n]] = tempreceive6[nl[n]] + (ref1_5 + ref2_5)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	tempreceive6_8[nl[n]] = tempreceive6[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#endif
	send1_fine[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_fine[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_fine[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* (NPR)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_fine[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* (NPR)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_fine[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)* (NPR)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_fine[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* (NPR)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_fine[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)* (NPR)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	receive6_fine[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)* (NPR)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3fine[nl[n]] = receive1_fine[nl[n]];
	receive1_4fine[nl[n]] = receive1_fine[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive1_7fine[nl[n]] = receive1_fine[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive1_8fine[nl[n]] = receive1_fine[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive2_1fine[nl[n]] = receive2_fine[nl[n]];
	receive2_2fine[nl[n]] = receive2_fine[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_3fine[nl[n]] = receive2_fine[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_4fine[nl[n]] = receive2_fine[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive3_1fine[nl[n]] = receive3_fine[nl[n]];
	receive3_2fine[nl[n]] = receive3_fine[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive3_5fine[nl[n]] = receive3_fine[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive3_6fine[nl[n]] = receive3_fine[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive4_5fine[nl[n]] = receive4_fine[nl[n]];
	receive4_6fine[nl[n]] = receive4_fine[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_7fine[nl[n]] = receive4_fine[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_8fine[nl[n]] = receive4_fine[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)	
	receive5_1fine[nl[n]] = receive5_fine[nl[n]];
	receive5_3fine[nl[n]] = receive5_fine[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_5fine[nl[n]] = receive5_fine[nl[n]] + (ref1_6 + ref2_6)*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_7fine[nl[n]] = receive5_fine[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive6_2fine[nl[n]] = receive6_fine[nl[n]];
	receive6_4fine[nl[n]] = receive6_fine[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_6fine[nl[n]] = receive6_fine[nl[n]] + (ref1_5 + ref2_5)*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_8fine[nl[n]] = receive6_fine[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#if(CPU_OPENMP)
	send1_flux[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_flux[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_flux[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive2_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	receive3_flux[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive4_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	receive5_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	receive6_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif	
	#if(N_LEVELS>1)
	receive1_3flux[nl[n]] = receive1_flux[nl[n]];
	receive1_4flux[nl[n]] = receive1_flux[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_7flux[nl[n]] = receive1_flux[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_8flux[nl[n]] = receive1_flux[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive2_1flux[nl[n]] = receive2_flux[nl[n]];
	receive2_2flux[nl[n]] = receive2_flux[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_3flux[nl[n]] = receive2_flux[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_4flux[nl[n]] = receive2_flux[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive3_1flux[nl[n]] = receive3_flux[nl[n]];
	receive3_2flux[nl[n]] = receive3_flux[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_5flux[nl[n]] = receive3_flux[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_6flux[nl[n]] = receive3_flux[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive4_5flux[nl[n]] = receive4_flux[nl[n]];
	receive4_6flux[nl[n]] = receive4_flux[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_7flux[nl[n]] = receive4_flux[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_8flux[nl[n]] = receive4_flux[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	receive5_1flux[nl[n]] = receive5_flux[nl[n]];
	receive5_3flux[nl[n]] = receive5_flux[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_5flux[nl[n]] = receive5_flux[nl[n]] + (ref2_6 + ref1_6)*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_7flux[nl[n]] = receive5_flux[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive6_2flux[nl[n]] = receive6_flux[nl[n]];
	receive6_4flux[nl[n]] = receive6_flux[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_6flux[nl[n]] = receive6_flux[nl[n]] + (ref2_5 + ref1_5)*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_8flux[nl[n]] = receive6_flux[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	#endif
	receive1_flux1[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive2_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	receive3_flux1[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive4_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	receive5_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	receive6_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif	
	#if(N_LEVELS>1)
	receive1_3flux1[nl[n]] = receive1_flux1[nl[n]];
	receive1_4flux1[nl[n]] = receive1_flux1[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_7flux1[nl[n]] = receive1_flux1[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_8flux1[nl[n]] = receive1_flux1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive2_1flux1[nl[n]] = receive2_flux1[nl[n]];
	receive2_2flux1[nl[n]] = receive2_flux1[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_3flux1[nl[n]] = receive2_flux1[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_4flux1[nl[n]] = receive2_flux1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive3_1flux1[nl[n]] = receive3_flux1[nl[n]];
	receive3_2flux1[nl[n]] = receive3_flux1[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_5flux1[nl[n]] = receive3_flux1[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_6flux1[nl[n]] = receive3_flux1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive4_5flux1[nl[n]] = receive4_flux1[nl[n]];
	receive4_6flux1[nl[n]] = receive4_flux1[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_7flux1[nl[n]] = receive4_flux1[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_8flux1[nl[n]] = receive4_flux1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	receive5_1flux1[nl[n]] = receive5_flux1[nl[n]];
	receive5_3flux1[nl[n]] = receive5_flux1[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_5flux1[nl[n]] = receive5_flux1[nl[n]] + (ref2_6 + ref1_6)*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_7flux1[nl[n]] = receive5_flux1[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive6_2flux1[nl[n]] = receive6_flux1[nl[n]];
	receive6_4flux1[nl[n]] = receive6_flux1[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_6flux1[nl[n]] = receive6_flux1[nl[n]] + (ref2_5 + ref1_5)*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_8flux1[nl[n]] = receive6_flux1[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	receive1_3flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_4flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_7flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_8flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive2_1flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_2flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_3flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_4flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive3_1flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_2flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_5flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_6flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive4_5flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_6flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_7flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_8flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	#if(N3G>0)
	receive5_1flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_3flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_5flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_7flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive6_2flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_4flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_6flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_8flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5) + 2 * N2G) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	#endif
	#endif
	#endif
	#if(CPU_OPENMP || 1)
	send1_E[nl[n]] = (double *)malloc(2 * (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_E[nl[n]] = (double *)malloc(2 * (BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_E[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_E[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_E[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_E[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_E[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_E[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3E[nl[n]] = receive1_E[nl[n]];
	receive1_4E[nl[n]] = receive1_E[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_7E[nl[n]] = receive1_E[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_8E[nl[n]] = receive1_E[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive2_1E[nl[n]] = receive2_E[nl[n]];
	receive2_2E[nl[n]] = receive2_E[nl[n]] + ref3_4*(2 * ((BS_2 + 2 * D2) / (1 + ref2_4))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_3E[nl[n]] = receive2_E[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_4E[nl[n]] = receive2_E[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive3_1E[nl[n]] = receive3_E[nl[n]];
	receive3_2E[nl[n]] = receive3_E[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_5E[nl[n]] = receive3_E[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_6E[nl[n]] = receive3_E[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive4_5E[nl[n]] = receive4_E[nl[n]];
	receive4_6E[nl[n]] = receive4_E[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_7E[nl[n]] = receive4_E[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_8E[nl[n]] = receive4_E[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	receive5_1E[nl[n]] = receive5_E[nl[n]];
	receive5_3E[nl[n]] = receive5_E[nl[n]] + ref2_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_5E[nl[n]] = receive5_E[nl[n]] + (ref2_6 + ref1_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_7E[nl[n]] = receive5_E[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive6_2E[nl[n]] = receive6_E[nl[n]];
	receive6_4E[nl[n]] = receive6_E[nl[n]] + ref2_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_6E[nl[n]] = receive6_E[nl[n]] + (ref2_5 + ref1_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_8E[nl[n]] = receive6_E[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	#endif
	receive1_E1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_E1[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_E1[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_E1[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_E1[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_E1[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3E1[nl[n]] = receive1_E1[nl[n]];
	receive1_4E1[nl[n]] = receive1_E1[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_7E1[nl[n]] = receive1_E1[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_8E1[nl[n]] = receive1_E1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive2_1E1[nl[n]] = receive2_E1[nl[n]];
	receive2_2E1[nl[n]] = receive2_E1[nl[n]] + ref3_4*(2 * ((BS_2 + 2 * D2) / (1 + ref2_4))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_3E1[nl[n]] = receive2_E1[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_4E1[nl[n]] = receive2_E1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive3_1E1[nl[n]] = receive3_E1[nl[n]];
	receive3_2E1[nl[n]] = receive3_E1[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_5E1[nl[n]] = receive3_E1[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_6E1[nl[n]] = receive3_E1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive4_5E1[nl[n]] = receive4_E1[nl[n]];
	receive4_6E1[nl[n]] = receive4_E1[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_7E1[nl[n]] = receive4_E1[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_8E1[nl[n]] = receive4_E1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	receive5_1E1[nl[n]] = receive5_E1[nl[n]];
	receive5_3E1[nl[n]] = receive5_E1[nl[n]] + ref2_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_5E1[nl[n]] = receive5_E1[nl[n]] + (ref2_6 + ref1_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_7E1[nl[n]] = receive5_E1[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive6_2E1[nl[n]] = receive6_E1[nl[n]];
	receive6_4E1[nl[n]] = receive6_E1[nl[n]] + ref2_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_6E1[nl[n]] = receive6_E1[nl[n]] + (ref2_5 + ref1_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_8E1[nl[n]] = receive6_E1[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	receive1_3E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_4E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_7E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_8E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive2_1E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_2E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_3E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_4E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive3_1E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_2E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_5E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_6E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive4_5E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_6E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_7E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_8E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive5_1E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_3E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_5E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_7E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_2E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_4E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_6E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_8E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#endif
	send_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	send_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive_E3_corn1_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#endif
	#endif
}

void free_arrays(int n){
	int i, count_node = 0, count_gpu = 0;

	//Count on node/GPU
	for (i = 0; i < NB_LOCAL; i++){
		if (mem_spot[i] != -1){
			count_node++; //Number of allocated blocks on node
			if (mem_spot_gpu[i] == block[n][AMR_GPU] && GPU_ENABLED==1) count_gpu++;
		}
	}
	if (count_gpu < (MAX_BLOCKS) || count_node < MAX_BLOCKS * N_GPU){
		mem_spot[nl[n]] = 0;
		free_bound_cpu(n);
		return;
	}
	else mem_spot[nl[n]] = -1;

	free(p[nl[n]]);
	free(ph[nl[n]]);
	#if(LEER)
	free(V[nl[n]]);
	#endif
	#if(STAGGERED)
	free(ps[nl[n]]);
	free(psh[nl[n]]);
	#endif
	free(dq[nl[n]]);
	free(F1[nl[n]]);
	free(F2[nl[n]]);
	free(F3[nl[n]]);
	free(pflag[nl[n]]);
	#if(GPU_DEBUG || CPU_OPENMP || 1)
	#if(STAGGERED)
	free(dE[nl[n]]);
	#endif
	free(E_corn[nl[n]]);
	#endif
	free(failimage[nl[n]]);
	free(conn[nl[n]]);
	free(gcov[nl[n]]);
	free(gcon[nl[n]]);
	free(gdet[nl[n]]);
	#if(HLLC)
	free(Mud[nl[n]]);
	free(Mud_inv[nl[n]]);
	#endif
	#if(ZIRI_DUMP)
	free(dump_buffer[nl[n]]);
	free(dxdxp_z[nl[n]]);
	free(dxpdx_z[nl[n]]);
	#endif
	#if (ELLIPTICAL2)
	free(dU_s[nl[n]]);
	#endif
	free(Katm[nl[n]]);
	free(array[nl[n]]);
	free(array_rdump[nl[n]]);
	free(array_gdump1[nl[n]]);
	free(array_gdump2[nl[n]]);
	free(array_diag[nl[n]]);
	free_bound_cpu(n);
}

void free_bound_cpu(int n){
	free(send1[nl[n]]);
	free(send2[nl[n]]);
	free(send3[nl[n]]);
	free(send4[nl[n]]);
	#if(N3G>0)
	free(send5[nl[n]]);
	free(send6[nl[n]]);
	#endif
	free(receive1[nl[n]]);
	#if(N_LEVELS>1)
	free(receive1_3[nl[n]]);
	free(receive1_4[nl[n]]);
	free(receive1_7[nl[n]]);
	free(receive1_8[nl[n]]);
	#endif
	free(receive2[nl[n]]);
	free(receive3[nl[n]]);
	#if(N_LEVELS>1)
	free(receive3_1[nl[n]]);
	free(receive3_2[nl[n]]);
	free(receive3_5[nl[n]]);
	free(receive3_6[nl[n]]);
	#endif
	free(receive4[nl[n]]);
	#if(N3G>0)
	free(receive5[nl[n]]);
	free(receive6[nl[n]]);
	#endif
	#if(PRESTEP==-100 || PRESTEP2==-100)
	free(tempreceive1[nl[n]]);
	#if(N_LEVELS>1)
	free(tempreceive1_3[nl[n]]);
	free(tempreceive1_4[nl[n]]);
	free(tempreceive1_7[nl[n]]);
	free(tempreceive1_8[nl[n]]);
	#endif
	free(tempreceive2[nl[n]]);
	free(tempreceive3[nl[n]]);
	#if(N_LEVELS>1)
	free(tempreceive3_1[nl[n]]);
	free(tempreceive3_2[nl[n]]);
	free(tempreceive3_5[nl[n]]);
	free(tempreceive3_6[nl[n]]);
	#endif
	free(tempreceive4[nl[n]]);
	#if(N3G>0)
	free(tempreceive5[nl[n]]);
	free(tempreceive6[nl[n]]);
	#endif

	#endif
	free(send1_fine[nl[n]]);
	free(send2_fine[nl[n]]);
	free(send3_fine[nl[n]]);
	free(send4_fine[nl[n]]);
	#if(N3G>0)
	free(send5_fine[nl[n]]);
	free(send6_fine[nl[n]]);
	#endif
	free(receive1_fine[nl[n]]);
	free(receive2_fine[nl[n]]);
	free(receive3_fine[nl[n]]);
	free(receive4_fine[nl[n]]);
	#if(N3G>0)
	free(receive5_fine[nl[n]]);
	free(receive6_fine[nl[n]]);
	#endif
	#if(CPU_OPENMP)
	free(send1_flux[nl[n]]);
	free(send2_flux[nl[n]]);
	free(send3_flux[nl[n]]);
	free(send4_flux[nl[n]]);
	#if(N3G>0)
	free(send5_flux[nl[n]]);
	free(send6_flux[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive1_flux[nl[n]]);
	free(receive2_flux[nl[n]]);
	free(receive3_flux[nl[n]]);
	free(receive4_flux[nl[n]]);
	free(receive5_flux[nl[n]]);
	free(receive6_flux[nl[n]]);
	free(receive1_flux1[nl[n]]);
	free(receive2_flux1[nl[n]]);
	free(receive3_flux1[nl[n]]);
	free(receive4_flux1[nl[n]]);
	free(receive5_flux1[nl[n]]);
	free(receive6_flux1[nl[n]]);
	free(receive1_3flux2[nl[n]]);
	free(receive1_4flux2[nl[n]]);
	free(receive1_7flux2[nl[n]]);
	free(receive1_8flux2[nl[n]]);
	free(receive2_1flux2[nl[n]]);
	free(receive2_2flux2[nl[n]]);
	free(receive2_3flux2[nl[n]]);
	free(receive2_4flux2[nl[n]]);
	free(receive3_1flux2[nl[n]]);
	free(receive3_2flux2[nl[n]]);
	free(receive3_5flux2[nl[n]]);
	free(receive3_6flux2[nl[n]]);
	free(receive4_5flux2[nl[n]]);
	free(receive4_6flux2[nl[n]]);
	free(receive4_7flux2[nl[n]]);
	free(receive4_8flux2[nl[n]]);
	#if(N3G>0)
	free(receive5_1flux2[nl[n]]);
	free(receive5_3flux2[nl[n]]);
	free(receive5_5flux2[nl[n]]);
	free(receive5_7flux2[nl[n]]);
	free(receive6_2flux2[nl[n]]);
	free(receive6_4flux2[nl[n]]);
	free(receive6_6flux2[nl[n]]);
	free(receive6_8flux2[nl[n]]);
	#endif
	#endif
	#endif
	#if(CPU_OPENMP || 1)
	free(send1_E[nl[n]]);
	free(send2_E[nl[n]]);
	free(send3_E[nl[n]]);
	free(send4_E[nl[n]]);
	#if(N3G>0)
	free(send5_E[nl[n]]);
	free(send6_E[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive1_E[nl[n]]);
	free(receive2_E[nl[n]]);
	free(receive3_E[nl[n]]);
	free(receive4_E[nl[n]]);
	free(receive5_E[nl[n]]);
	free(receive6_E[nl[n]]);
	free(receive1_E1[nl[n]]);
	free(receive2_E1[nl[n]]);
	free(receive3_E1[nl[n]]);
	free(receive4_E1[nl[n]]);
	free(receive5_E1[nl[n]]);
	free(receive6_E1[nl[n]]);
	free(receive1_3E2[nl[n]]);
	free(receive1_4E2[nl[n]]);
	free(receive1_7E2[nl[n]]);
	free(receive1_8E2[nl[n]]);
	free(receive2_1E2[nl[n]]);
	free(receive2_2E2[nl[n]]);
	free(receive2_3E2[nl[n]]);
	free(receive2_4E2[nl[n]]);
	free(receive3_1E2[nl[n]]);
	free(receive3_2E2[nl[n]]);
	free(receive3_5E2[nl[n]]);
	free(receive3_6E2[nl[n]]);
	free(receive4_5E2[nl[n]]);
	free(receive4_6E2[nl[n]]);
	free(receive4_7E2[nl[n]]);
	free(receive4_8E2[nl[n]]);
	#if(N3G>0)
	free(receive5_1E2[nl[n]]);
	free(receive5_3E2[nl[n]]);
	free(receive5_5E2[nl[n]]);
	free(receive5_7E2[nl[n]]);
	free(receive6_2E2[nl[n]]);
	free(receive6_4E2[nl[n]]);
	free(receive6_6E2[nl[n]]);
	free(receive6_8E2[nl[n]]);
	#endif
	#endif
	free(send_E3_corn1[nl[n]]);
	free(send_E3_corn2[nl[n]]);
	free(send_E3_corn3[nl[n]]);
	free(send_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(send_E2_corn5[nl[n]]);
	free(send_E2_corn6[nl[n]]);
	free(send_E2_corn7[nl[n]]);
	free(send_E2_corn8[nl[n]]);
	free(send_E1_corn9[nl[n]]);
	free(send_E1_corn10[nl[n]]);
	free(send_E1_corn11[nl[n]]);
	free(send_E1_corn12[nl[n]]);
	#endif
	free(receive_E3_corn1[nl[n]]);
	free(receive_E3_corn2[nl[n]]);
	free(receive_E3_corn3[nl[n]]);
	free(receive_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5[nl[n]]);
	free(receive_E2_corn6[nl[n]]);
	free(receive_E2_corn7[nl[n]]);
	free(receive_E2_corn8[nl[n]]);
	free(receive_E1_corn9[nl[n]]);
	free(receive_E1_corn10[nl[n]]);
	free(receive_E1_corn11[nl[n]]);
	free(receive_E1_corn12[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive_E3_corn1_1[nl[n]]);
	free(receive_E3_corn2_1[nl[n]]);
	free(receive_E3_corn3_1[nl[n]]);
	free(receive_E3_corn4_1[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_1[nl[n]]);
	free(receive_E2_corn6_1[nl[n]]);
	free(receive_E2_corn7_1[nl[n]]);
	free(receive_E2_corn8_1[nl[n]]);
	free(receive_E1_corn9_1[nl[n]]);
	free(receive_E1_corn10_1[nl[n]]);
	free(receive_E1_corn11_1[nl[n]]);
	free(receive_E1_corn12_1[nl[n]]);
	#endif
	free(receive_E3_corn1_2[nl[n]]);
	free(receive_E3_corn2_2[nl[n]]);
	free(receive_E3_corn3_2[nl[n]]);
	free(receive_E3_corn4_2[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_2[nl[n]]);
	free(receive_E2_corn6_2[nl[n]]);
	free(receive_E2_corn7_2[nl[n]]);
	free(receive_E2_corn8_2[nl[n]]);
	free(receive_E1_corn9_2[nl[n]]);
	free(receive_E1_corn10_2[nl[n]]);
	free(receive_E1_corn11_2[nl[n]]);
	free(receive_E1_corn12_2[nl[n]]);
	#endif
	#endif
	free(tempreceive_E3_corn1[nl[n]]);
	free(tempreceive_E3_corn2[nl[n]]);
	free(tempreceive_E3_corn3[nl[n]]);
	free(tempreceive_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5[nl[n]]);
	free(tempreceive_E2_corn6[nl[n]]);
	free(tempreceive_E2_corn7[nl[n]]);
	free(tempreceive_E2_corn8[nl[n]]);
	free(tempreceive_E1_corn9[nl[n]]);
	free(tempreceive_E1_corn10[nl[n]]);
	free(tempreceive_E1_corn11[nl[n]]);
	free(tempreceive_E1_corn12[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(tempreceive_E3_corn1_1[nl[n]]);
	free(tempreceive_E3_corn2_1[nl[n]]);
	free(tempreceive_E3_corn3_1[nl[n]]);
	free(tempreceive_E3_corn4_1[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_1[nl[n]]);
	free(tempreceive_E2_corn6_1[nl[n]]);
	free(tempreceive_E2_corn7_1[nl[n]]);
	free(tempreceive_E2_corn8_1[nl[n]]);
	free(tempreceive_E1_corn9_1[nl[n]]);
	free(tempreceive_E1_corn10_1[nl[n]]);
	free(tempreceive_E1_corn11_1[nl[n]]);
	free(tempreceive_E1_corn12_1[nl[n]]);
	#endif
	free(tempreceive_E3_corn1_2[nl[n]]);
	free(tempreceive_E3_corn2_2[nl[n]]);
	free(tempreceive_E3_corn3_2[nl[n]]);
	free(tempreceive_E3_corn4_2[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_2[nl[n]]);
	free(tempreceive_E2_corn6_2[nl[n]]);
	free(tempreceive_E2_corn7_2[nl[n]]);
	free(tempreceive_E2_corn8_2[nl[n]]);
	free(tempreceive_E1_corn9_2[nl[n]]);
	free(tempreceive_E1_corn10_2[nl[n]]);
	free(tempreceive_E1_corn11_2[nl[n]]);
	free(tempreceive_E1_corn12_2[nl[n]]);
	#endif
	free(receive_E3_corn1_12[nl[n]]);
	free(receive_E3_corn2_12[nl[n]]);
	free(receive_E3_corn3_12[nl[n]]);
	free(receive_E3_corn4_12[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_12[nl[n]]);
	free(receive_E2_corn6_12[nl[n]]);
	free(receive_E2_corn7_12[nl[n]]);
	free(receive_E2_corn8_12[nl[n]]);
	free(receive_E1_corn9_12[nl[n]]);
	free(receive_E1_corn10_12[nl[n]]);
	free(receive_E1_corn11_12[nl[n]]);
	free(receive_E1_corn12_12[nl[n]]);
	#endif
	free(receive_E3_corn1_22[nl[n]]);
	free(receive_E3_corn2_22[nl[n]]);
	free(receive_E3_corn3_22[nl[n]]);
	free(receive_E3_corn4_22[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_22[nl[n]]);
	free(receive_E2_corn6_22[nl[n]]);
	free(receive_E2_corn7_22[nl[n]]);
	free(receive_E2_corn8_22[nl[n]]);
	free(receive_E1_corn9_22[nl[n]]);
	free(receive_E1_corn10_22[nl[n]]);
	free(receive_E1_corn11_22[nl[n]]);
	free(receive_E1_corn12_22[nl[n]]);
	#endif
	#endif
	#endif
}

int index_3D(int n, int i, int j, int z)
{
	return(((i - N1_GPU_offset[n]) + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(BS_3 + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
}
int index_2D(int n, int i, int j, int z)
{
	#if(!NSY)
	return(((i - N1_GPU_offset[n]) + N1G)*(BS_2 + 2 * N2G) + ((j - N2_GPU_offset[n]) + N2G));
	#else
	return(((i - N1_GPU_offset[n]) + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(BS_3 + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
	#endif
}

/*****************************************************************/
/*****************************************************************
  set_grid():
  ----------

       -- calculates all grid functions that remain constant 
          over time, such as the metric (gcov), inverse metric 
          (gcon), connection coefficients (conn), and sqrt of 
          the metric's determinant (gdet).

 *****************************************************************/
void set_grid(int n)
{
	int i,j,z,k,i1,j1,z1,zsize=1,zlevel=0,zoffset=0 ;
	double r, th, phi;
	struct of_geom geom ;

	/* set up boundaries, steps in coordinate grid */
	set_points(n) ;
	dV = dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3];
	double X[NDIM];

	double temp = a;
	#pragma omp parallel private(X,i,j,z,k,geom, i1,j1,z1,r,th,phi,a,zsize,zlevel,zoffset)
	{
		DLOOPA X[j] = 0.;
		#pragma omp for collapse(2) schedule(static,(BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)
		#if(!NSY)
		ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]) {
		#else
		ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		#endif
			if (j<0 || j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && TRANS_BOUND) a = -temp;
			else a = temp;

			zlevel = 0;
			if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && j < N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n]) + D2))) / log(2.)), N_LEVELS_1D_INT);
			if ((block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) && j >= N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
			zsize = (int)pow(2.0, (double)zlevel);
			zoffset = (z - N3_GPU_offset[n]) % zsize;

			/* zone-centered */
			coord(n, i, j, z - zoffset + zsize / 2, CENT, X);
			gcov_func(X, gcov[nl[n]][index_2D(n, i, j, z)][CENT]);
			gdet[nl[n]][index_2D(n, i, j, z)][CENT] = gdet_func(gcov[nl[n]][index_2D(n, i, j, z)][CENT]);
			if (j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2])-1 && TRANS_BOUND == 1)gdet[nl[n]][index_2D(n, i, j, z)][CENT] *= 1.0;
			gcon_func(gcov[nl[n]][index_2D(n, i, j, z)][CENT], gcon[nl[n]][index_2D(n, i, j, z)][CENT]);
			get_geometry(n, i, j, z, CENT, &geom);
			conn_func(X, &geom, conn[nl[n]][index_2D(n, i, j, z)]);
			if ((j == -1 || j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2])) && (TRANS_BOUND==1)){
				//for (i1 = 0; i1 < NDIM; i1++)for (j1 = 0; j1 < NDIM; j1++)for (z1 = 0; z1 < NDIM; z1++)conn[nl[n]][index_2D(n, i, j, z)][i1][j1][z1] = 0.;
			}

			/* r-face-centered */
			coord(n, i, j, z - zoffset + zsize / 2, FACE1, X);
			gcov_func(X, gcov[nl[n]][index_2D(n, i, j, z)][FACE1]);
			gdet[nl[n]][index_2D(n, i, j, z)][FACE1] = gdet_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE1]);
			gcon_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE1], gcon[nl[n]][index_2D(n, i, j, z)][FACE1]);
			
			/* phi-face-centered */
			coord(n, i, j, z - zoffset, FACE3, X);
			gcov_func(X, gcov[nl[n]][index_2D(n, i, j, z)][FACE3]);
			gdet[nl[n]][index_2D(n, i, j, z)][FACE3] = gdet_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE3]);
			gcon_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE3], gcon[nl[n]][index_2D(n, i, j, z)][FACE3]);

			/* theta-face-centered */
			if (j == 0 && TRANS_BOUND==1){
				//coord(n, i, 1, z, FACE2, X);
				a = 0. ;
				coord(n, i, j, z - zoffset + zsize / 2, FACE2, X);
			}
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && TRANS_BOUND==1){
				//coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z, FACE2, X);
				coord(n, i, j, z - zoffset + zsize / 2, FACE2, X);
				a = 0.;
			}
			else coord(n, i, j, z - zoffset + zsize / 2, FACE2, X);
			gcov_func(X, gcov[nl[n]][index_2D(n, i, j, z)][FACE2]);
			gdet[nl[n]][index_2D(n, i, j, z)][FACE2] = gdet_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE2]);
			gcon_func(gcov[nl[n]][index_2D(n, i, j, z)][FACE2], gcon[nl[n]][index_2D(n, i, j, z)][FACE2]);	
		}
	}
	#if(HLLC)
	set_Mud(n);
	#endif
	#if(LEER)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		//Set temporary array with r, th, phi distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		coord(n, i, j, z, FACE1, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][0] = r;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][3] = r;

		coord(n, i, j, z, FACE2, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][1] = th;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][4] = th;

		coord(n, i, j, z, FACE3, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][2] = phi;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[nl[n]][index_3D(n, i, j, z)][5] = phi;

		for (k = 0; k < 6; k++) V[nl[n]][index_3D(n, i, j, z)][k] = 0.0;
	}
	ZSLOOP3D(-N1G + N1_GPU_offset[n],-N1G + N1_GPU_offset[n], -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		V[nl[n]][index_3D(n, i, j, z)][3] = V[nl[n]][index_3D(n, i, j, z)][0] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE1][1][1]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], -N2G + N2_GPU_offset[n], -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		V[nl[n]][index_3D(n, i, j, z)][4] = V[nl[n]][index_3D(n, i, j, z)][1] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE2][2][2]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], -N3G + N3_GPU_offset[n]) {
		V[nl[n]][index_3D(n, i, j, z)][5] = V[nl[n]][index_3D(n, i, j, z)][2] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE3][3][3]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1+ N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		//Calculate distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		V[nl[n]][index_3D(n, i, j, z)][0] = V[nl[n]][index_3D(n, i - D1, j, z)][3] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i - D1, j, z)][CENT][1][1]);
		V[nl[n]][index_3D(n, i, j, z)][3] = V[nl[n]][index_3D(n, i, j, z)][0] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE1][1][1]);
		
		V[nl[n]][index_3D(n, i, j, z)][1] = V[nl[n]][index_3D(n, i, j - D2, z)][4] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j - D2, z)][CENT][2][2]);
		V[nl[n]][index_3D(n, i, j, z)][4] = V[nl[n]][index_3D(n, i, j, z)][1] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE2][2][2]);

		V[nl[n]][index_3D(n, i, j, z)][2] = V[nl[n]][index_3D(n, i, j, z - D3)][5] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z - D3)][CENT][3][3]);
		V[nl[n]][index_3D(n, i, j, z)][5] = V[nl[n]][index_3D(n, i, j, z)][2] + 0.5*sqrt(gcov[nl[n]][index_2D(n, i, j, z)][FACE3][3][3]);
	}
	#endif

	a=temp;

	#if ZIRI_DUMP
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		coord(n,i, j, z, CENT, X);
		dxdxp_func(X, dxdxp_z[nl[n]][index_3D(n ,i,j,z)]);
		//invert_matrix(dxdxp_z[nl[n]][index_3D(n ,i,j,z)], dxpdx_z[nl[n]][index_3D(n ,i,j,z)]);
	}
	#endif

	/* done! */
}

double get_wall_time(){
	#ifdef __unix__   
	struct timeval time;
	if (gettimeofday(&time, NULL)){
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
	#else
	return clock() / CLOCKS_PER_SEC;
	#endif
}