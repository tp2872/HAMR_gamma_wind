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
	int i, j, u, n;
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
		init();
		#if(DEREFINE_POLE)
		derefine_pole();
		#endif
	}

	/* do initial diagnostics */
	first_dump = 0;
	diag(INIT_OUT);

	DTl = 20.0;
	tdump = t + DTd;
	tlog = t + DTl;
	tref = t + TREF;
	time_spent3 = 0.0;
	begin1 = time(NULL);
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
		if (nstep % (20 * AMR_SWITCHTIMELEVEL) == 0){
			end1 = time(NULL);
			#if (GPU_ENABLED==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			bound_prim(p, 1);
			if (dt > 0.5) break;
		}

		//Refine every TREF
		if (t >= tref && nstep % (20 * AMR_SWITCHTIMELEVEL) == 0) {
			#if(!DEREFINE_POLE)
			check_refcrit();
			if (rank == 0) fprintf(stderr, "Refinement succesfull! \n");
			#endif
			tref += TREF;
		}

		/* Put out dump file*/
		if (t >= tdump && nstep % (20 * AMR_SWITCHTIMELEVEL) == 0) {
			diag(DUMP_OUT) ;
			tdump += DTd;
		}

		//Put out log file and rdump file
		if (t >= tlog && nstep % (20 * AMR_SWITCHTIMELEVEL) == 0) {
			//restart_write(); //do restart dumb simultaneous with log
			tlog +=  DTl;
		}			
		
		#if TIMER
		if (nstep % (20*AMR_SWITCHTIMELEVEL) == 0){
			diag(LOG_OUT);
			MPI_Allreduce(MPI_IN_PLACE, &ndt1, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt2, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt3, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			if (rank == 0){
				fprintf(stderr, "Runtime: %f MPI-time: %f ", (double)(end1 - begin1), time_spent3);
				fprintf(stderr, "dt1: %f dt2: %f dt3: %f nstep: %d \n", ndt1,ndt2,ndt3,nstep);
				fflush(stderr);
			}
			time_spent3 = 0.0;	

			//Safe and exit at end of 24 hour runtime
			if ((double)(begin2 - end1) > 24.*3600.){
				restart_write();
				//break;
			}
			begin1 = time(NULL);			
		}
		#endif
	}
	//cuProfilerStop();

	/* do final diagnostics */
	diag(FINAL_OUT) ;

	/*Close GPU*/
	for (n=0; n<n_active; n++) GPU_finish(n_ord[n]);
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
	//#if(GPU_ENABLED)
	cudaGetDeviceCount(&numdevices);
	cudaSetDevice(local_rank%numdevices);
	//#endif
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
	int i;
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
	array[n] = (float *)calloc(9 * BS_1*BS_2*BS_3, sizeof(float));
	array_gdump1[n] = (double *)calloc(9 * BS_1*BS_2*BS_3, sizeof(double));
	array_gdump2[n] = (double *)calloc(49 * BS_1*BS_2, sizeof(double));
	array_rdump[n] = (double *)calloc((NPR + NDIM) * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double));
	array_diag[n] = (float *)calloc(4 * BS_1*BS_2*BS_3, sizeof(float));
	Katm[n] = (double(*))calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	p[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NPR]));
	ph[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NPR]));
	#if(STAGGERED)
	ps[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NDIM]));
	psh[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NDIM]));
	#endif
	#if(LEER)
	V[n] = (double(*)[6])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double[6]));
	#endif
	dq[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	F1[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NPR]));
	F2[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NPR]));
	F3[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NPR]));
	pflag[n] = (int(*))calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(int));
	#if(CPU_OPENMP)
	#if(STAGGERED)
	dE[n] = (double(*)[2][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[2][NDIM][NDIM]));
	#endif
	E_corn[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double[NDIM]));
	#endif
	failimage[n] = (int(*)[NFAIL])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(int[NFAIL]));
	#if(!NONSYMMETRIC)
	conn[n] = (double(*)[NDIM][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G), sizeof(double[NDIM][NDIM][NDIM]));
	gcov[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G), sizeof(double[NPG][NDIM][NDIM]));
	gcon[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G), sizeof(double[NPG][NDIM][NDIM]));
	gdet[n] = (double(*)[NPG])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G), sizeof(double[NPG]));
	#else
	conn[n] = (double(*)[NDIM][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NDIM][NDIM][NDIM]));
	gcov[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NPG][NDIM][NDIM]));
	gcon[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPG][NDIM][NDIM]));
	gdet[n] = (double(*)[NPG])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPG]));
	#endif
	#if(ZIRI_DUMP)
	dump_buffer[n] = (double(*))calloc(N1_GPU[n] * N2_GPU[n] * N3_GPU[n] * 13 *sizeof(double));
	dxdxp_z[n] = (double(*)[NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NDIM][NDIM]));
	dxpdx_z[n] = (double(*)[NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NDIM][NDIM]));
	#endif
	#if (ELLIPTICAL2)
	dU_s[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G),sizeof(double[NPR]));
	#endif
	receive1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	tempreceive1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	tempreceive2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	tempreceive3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	tempreceive4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	tempreceive5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	tempreceive6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	send1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	send5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) * (N1_GPU[n] + 2 * N1G), sizeof(double));
	send6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) * (N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	#if(N_LEVELS>1)
	send1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	send5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	send6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	#endif
	receive1_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	send1_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	send5_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	send6_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	receive1_3fine[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_4fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_7fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_8fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_1fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_2fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_3fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_4fine[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_1fine[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_2fine[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_5fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_6fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_5fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_6fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_7fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_8fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive5_1fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	#if(CPU_OPENMP)
	receive1_flux[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3_flux[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	receive1_flux1[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3_flux1[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6_flux1[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3flux[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4flux[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1flux[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2flux[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	receive1_3flux1[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4flux1[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1flux1[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2flux1[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	receive1_3flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	receive1_3E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	receive1_3E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	receive1_3E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_4E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_7E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive1_8E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_1E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_2E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_3E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive2_4E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_1E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_2E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_5E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive3_6E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_5E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_6E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_7E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	receive4_8E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_1E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	#endif
	send1_flux[n] = (double *)calloc(NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3_flux[n] = (double *)calloc(NPR*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	send5_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	send6_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	receive1_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	receive1_E1[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive2_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive3_E1[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	receive4_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	receive5_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	receive6_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	send1_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) , sizeof(double));
	#if(N3G>0)
	send5_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	send6_E[n] = (double *)calloc(2*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) , sizeof(double));
	#endif
	
	send_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	send_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	send_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	send_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	send_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	send_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	send_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	send_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	send_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	send_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	send_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	send_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	receive_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	tempreceive_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive_E3_corn1_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn2_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn3_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn4_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn6_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn7_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn8_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E1_corn9_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn10_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn11_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn12_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	receive_E3_corn1_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn2_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn3_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn4_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn6_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn7_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn8_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E1_corn9_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn10_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn11_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn12_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	tempreceive_E3_corn1_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn2_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn3_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn4_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn6_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn7_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn8_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E1_corn9_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn10_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn11_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn12_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	tempreceive_E3_corn1_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn2_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn3_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive_E3_corn4_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn6_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn7_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E2_corn8_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	tempreceive_E1_corn9_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn10_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn11_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive_E1_corn12_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	receive_E3_corn1_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn2_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn3_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn4_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn6_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn7_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn8_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E1_corn9_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn10_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn11_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn12_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	receive_E3_corn1_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn2_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn3_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	receive_E3_corn4_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn6_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn7_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E2_corn8_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G), sizeof(double));
	receive_E1_corn9_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn10_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn11_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	receive_E1_corn12_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	#endif
	#endif
	//#endif
}

void free_arrays(int n)
{
	free(p[n]);
	free(ph[n]);
	#if(LEER)
	free(V[n]);
	#endif
	#if(STAGGERED)
	free(ps[n]);
	free(psh[n]);
	#endif
	free(dq[n]);
	free(F1[n]);
	free(F2[n]);
	free(F3[n]);
	free(pflag[n]);
	#if(GPU_DEBUG || CPU_OPENMP)
	#if(STAGGERED)
	free(dE[n]);
	#endif
	free(E_corn[n]);
	#endif
	free(failimage[n]);
	free(conn[n]);
	free(gcov[n]);
	free(gcon[n]);
	free(gdet[n]);
	#if(ZIRI_DUMP)
	free(dump_buffer[n]);
	free(dxdxp_z[n]);
	free(dxpdx_z[n]);
	#endif
	#if (ELLIPTICAL2)
	free(dU_s[n]);
	#endif
	free(receive1[n]);
	free(receive2[n]);
	free(receive3[n]);
	free(receive4[n]);
	#if(N3G>0)
	free(receive5[n]);
	free(receive6[n]);
	#endif
	free(tempreceive1[n]);
	free(tempreceive2[n]);
	free(tempreceive3[n]);
	free(tempreceive4[n]);
	#if(N3G>0)
	free(tempreceive5[n]);
	free(tempreceive6[n]);
	#endif
	free(send1[n]);
	free(send2[n]);
	free(send3[n]);
	free(send4[n]);
	#if(N3G>0)
	free(send5[n]);
	free(send6[n]);
	#endif
	free(send1_3[n]);
	free(send1_4[n]);
	free(send1_7[n]);
	free(send1_8[n]);
	free(send2_1[n]);
	free(send2_2[n]);
	free(send2_3[n]);
	free(send2_4[n]);
	free(send3_1[n]);
	free(send3_2[n]);
	free(send3_5[n]);
	free(send3_6[n]);
	free(send4_5[n]);
	free(send4_6[n]);
	free(send4_7[n]);
	free(send4_8[n]);
	#if(N3G>0)
	free(send5_1[n]);
	free(send5_3[n]);
	free(send5_5[n]);
	free(send5_7[n]);
	free(send6_2[n]);
	free(send6_4[n]);
	free(send6_6[n]);
	free(send6_8[n]);
	#endif
	free(receive1_3[n]);
	free(receive1_4[n]);
	free(receive1_7[n]);
	free(receive1_8[n]);
	free(receive2_1[n]);
	free(receive2_2[n]);
	free(receive2_3[n]);
	free(receive2_4[n]);
	free(receive3_1[n]);
	free(receive3_2[n]);
	free(receive3_5[n]);
	free(receive3_6[n]);
	free(receive4_5[n]);
	free(receive4_6[n]);
	free(receive4_7[n]);
	free(receive4_8[n]);
	#if(N3G>0)
	free(receive5_1[n]);
	free(receive5_3[n]);
	free(receive5_5[n]);
	free(receive5_7[n]);
	free(receive6_2[n]);
	free(receive6_4[n]);
	free(receive6_6[n]);
	free(receive6_8[n]);
	#endif
	free(tempreceive1_3[n]);
	free(tempreceive1_4[n]);
	free(tempreceive1_7[n]);
	free(tempreceive1_8[n]);
	free(tempreceive2_1[n]);
	free(tempreceive2_2[n]);
	free(tempreceive2_3[n]);
	free(tempreceive2_4[n]);
	free(tempreceive3_1[n]);
	free(tempreceive3_2[n]);
	free(tempreceive3_5[n]);
	free(tempreceive3_6[n]);
	free(tempreceive4_5[n]);
	free(tempreceive4_6[n]);
	free(tempreceive4_7[n]);
	free(tempreceive4_8[n]);
	#if(N3G>0)
	free(tempreceive5_1[n]);
	free(tempreceive5_3[n]);
	free(tempreceive5_5[n]);
	free(tempreceive5_7[n]);
	free(tempreceive6_2[n]);
	free(tempreceive6_4[n]);
	free(tempreceive6_6[n]);
	free(tempreceive6_8[n]);
	#endif
	free(receive1_fine[n]);
	free(receive2_fine[n]);
	free(receive3_fine[n]);
	free(receive4_fine[n]);
	#if(N3G>0)
	free(receive5_fine[n]);
	free(receive6_fine[n]);
	#endif
	free(send1_fine[n]);
	free(send2_fine[n]);
	free(send3_fine[n]);
	free(send4_fine[n]);
	#if(N3G>0)
	free(send5_fine[n]);
	free(send6_fine[n]);
	#endif
	free(receive1_3fine[n]);
	free(receive1_4fine[n]);
	free(receive1_7fine[n]);
	free(receive1_8fine[n]);
	free(receive2_1fine[n]);
	free(receive2_2fine[n]);
	free(receive2_3fine[n]);
	free(receive2_4fine[n]);
	free(receive3_1fine[n]);
	free(receive3_2fine[n]);
	free(receive3_5fine[n]);
	free(receive3_6fine[n]);
	free(receive4_5fine[n]);
	free(receive4_6fine[n]);
	free(receive4_7fine[n]);
	free(receive4_8fine[n]);
	#if(N3G>0)
	free(receive5_1fine[n]);
	free(receive5_3fine[n]);
	free(receive5_5fine[n]);
	free(receive5_7fine[n]);
	free(receive6_2fine[n]);
	free(receive6_4fine[n]);
	free(receive6_6fine[n]);
	free(receive6_8fine[n]);
	#endif
	#if(CPU_OPENMP)
	free(receive1_flux[n]);
	free(receive2_flux[n]);
	free(receive3_flux[n]);
	free(receive4_flux[n]);
	free(receive5_flux[n]);
	free(receive6_flux[n]);
	free(receive1_3flux[n]);
	free(receive1_4flux[n]);
	free(receive1_7flux[n]);
	free(receive1_8flux[n]);
	free(receive2_1flux[n]);
	free(receive2_2flux[n]);
	free(receive2_3flux[n]);
	free(receive2_4flux[n]);
	free(receive3_1flux[n]);
	free(receive3_2flux[n]);
	free(receive3_5flux[n]);
	free(receive3_6flux[n]);
	free(receive4_5flux[n]);
	free(receive4_6flux[n]);
	free(receive4_7flux[n]);
	free(receive4_8flux[n]);
	#if(N3G>0)
	free(receive5_1flux[n]);
	free(receive5_3flux[n]);
	free(receive5_5flux[n]);
	free(receive5_7flux[n]);
	free(receive6_2flux[n]);
	free(receive6_4flux[n]);
	free(receive6_6flux[n]);
	free(receive6_8flux[n]);
	#endif
	free(receive1_flux1[n]);
	free(receive2_flux1[n]);
	free(receive3_flux1[n]);
	free(receive4_flux1[n]);
	free(receive5_flux1[n]);
	free(receive6_flux1[n]);
	free(receive1_3flux1[n]);
	free(receive1_4flux1[n]);
	free(receive1_7flux1[n]);
	free(receive1_8flux1[n]);
	free(receive2_1flux1[n]);
	free(receive2_2flux1[n]);
	free(receive2_3flux1[n]);
	free(receive2_4flux1[n]);
	free(receive3_1flux1[n]);
	free(receive3_2flux1[n]);
	free(receive3_5flux1[n]);
	free(receive3_6flux1[n]);
	free(receive4_5flux1[n]);
	free(receive4_6flux1[n]);
	free(receive4_7flux1[n]);
	free(receive4_8flux1[n]);
	#if(N3G>0)
	free(receive5_1flux1[n]);
	free(receive5_3flux1[n]);
	free(receive5_5flux1[n]);
	free(receive5_7flux1[n]);
	free(receive6_2flux1[n]);
	free(receive6_4flux1[n]);
	free(receive6_6flux1[n]);
	free(receive6_8flux1[n]);
	#endif
	free(receive1_3flux2[n]);
	free(receive1_4flux2[n]);
	free(receive1_7flux2[n]);
	free(receive1_8flux2[n]);
	free(receive2_1flux2[n]);
	free(receive2_2flux2[n]);
	free(receive2_3flux2[n]);
	free(receive2_4flux2[n]);
	free(receive3_1flux2[n]);
	free(receive3_2flux2[n]);
	free(receive3_5flux2[n]);
	free(receive3_6flux2[n]);
	free(receive4_5flux2[n]);
	free(receive4_6flux2[n]);
	free(receive4_7flux2[n]);
	free(receive4_8flux2[n]);
	#if(N3G>0)
	free(receive5_1flux2[n]);
	free(receive5_3flux2[n]);
	free(receive5_5flux2[n]);
	free(receive5_7flux2[n]);
	free(receive6_2flux2[n]);
	free(receive6_4flux2[n]);
	free(receive6_6flux2[n]);
	free(receive6_8flux2[n]);
	#endif
	free(send1_flux[n]);
	free(send2_flux[n]);
	free(send3_flux[n]);
	free(send4_flux[n]);
	#if(N3G>0)
	free(send5_flux[n]);
	free(send6_flux[n]);
	#endif
	free(receive1_E[n]);
	free(receive2_E[n]);
	free(receive3_E[n]);
	free(receive4_E[n]);
	free(receive5_E[n]);
	free(receive6_E[n]);
	free(receive1_3E[n]);
	free(receive1_4E[n]);
	free(receive1_7E[n]);
	free(receive1_8E[n]);
	free(receive2_1E[n]);
	free(receive2_2E[n]);
	free(receive2_3E[n]);
	free(receive2_4E[n]);
	free(receive3_1E[n]);
	free(receive3_2E[n]);
	free(receive3_5E[n]);
	free(receive3_6E[n]);
	free(receive4_5E[n]);
	free(receive4_6E[n]);
	free(receive4_7E[n]);
	free(receive4_8E[n]);
	#if(N3G>0)
	free(receive5_1E[n]);
	free(receive5_3E[n]);
	free(receive5_5E[n]);
	free(receive5_7E[n]);
	free(receive6_2E[n]);
	free(receive6_4E[n]);
	free(receive6_6E[n]);
	free(receive6_8E[n]);
	#endif
	free(receive1_E1[n]);
	free(receive2_E1[n]);
	free(receive3_E1[n]);
	free(receive4_E1[n]);
	free(receive5_E1[n]);
	free(receive6_E1[n]);
	free(receive1_3E1[n]);
	free(receive1_4E1[n]);
	free(receive1_7E1[n]);
	free(receive1_8E1[n]);
	free(receive2_1E1[n]);
	free(receive2_2E1[n]);
	free(receive2_3E1[n]);
	free(receive2_4E1[n]);
	free(receive3_1E1[n]);
	free(receive3_2E1[n]);
	free(receive3_5E1[n]);
	free(receive3_6E1[n]);
	free(receive4_5E1[n]);
	free(receive4_6E1[n]);
	free(receive4_7E1[n]);
	free(receive4_8E1[n]);
#if(N3G>0)
	free(receive5_1E1[n]);
	free(receive5_3E1[n]);
	free(receive5_5E1[n]);
	free(receive5_7E1[n]);
	free(receive6_2E1[n]);
	free(receive6_4E1[n]);
	free(receive6_6E1[n]);
	free(receive6_8E1[n]);
#endif
	free(receive1_3E2[n]);
	free(receive1_4E2[n]);
	free(receive1_7E2[n]);
	free(receive1_8E2[n]);
	free(receive2_1E2[n]);
	free(receive2_2E2[n]);
	free(receive2_3E2[n]);
	free(receive2_4E2[n]);
	free(receive3_1E2[n]);
	free(receive3_2E2[n]);
	free(receive3_5E2[n]);
	free(receive3_6E2[n]);
	free(receive4_5E2[n]);
	free(receive4_6E2[n]);
	free(receive4_7E2[n]);
	free(receive4_8E2[n]);
#if(N3G>0)
	free(receive5_1E2[n]);
	free(receive5_3E2[n]);
	free(receive5_5E2[n]);
	free(receive5_7E2[n]);
	free(receive6_2E2[n]);
	free(receive6_4E2[n]);
	free(receive6_6E2[n]);
	free(receive6_8E2[n]);
#endif
	free(send1_E[n]);
	free(send2_E[n]);
	free(send3_E[n]);
	free(send4_E[n]);
#if(N3G>0)
	free(send5_E[n]);
	free(send6_E[n]);
#endif
	free(send_E3_corn1[n]);
	free(send_E3_corn2[n]);
	free(send_E3_corn3[n]);
	free(send_E3_corn4[n]);
#if(N3G>0)
	free(send_E2_corn5[n]);
	free(send_E2_corn6[n]);
	free(send_E2_corn7[n]);
	free(send_E2_corn8[n]);
	free(send_E1_corn9[n]);
	free(send_E1_corn10[n]);
	free(send_E1_corn11[n]);
	free(send_E1_corn12[n]);
#endif
	free(receive_E3_corn1[n]);
	free(receive_E3_corn2[n]);
	free(receive_E3_corn3[n]);
	free(receive_E3_corn4[n]);
#if(N3G>0)
	free(receive_E2_corn5[n]);
	free(receive_E2_corn6[n]);
	free(receive_E2_corn7[n]);
	free(receive_E2_corn8[n]);
	free(receive_E1_corn9[n]);
	free(receive_E1_corn10[n]);
	free(receive_E1_corn11[n]);
	free(receive_E1_corn12[n]);
#endif
	free(receive_E3_corn1_1[n]);
	free(receive_E3_corn2_1[n]);
	free(receive_E3_corn3_1[n]);
	free(receive_E3_corn4_1[n]);
#if(N3G>0)
	free(receive_E2_corn5_1[n]);
	free(receive_E2_corn6_1[n]);
	free(receive_E2_corn7_1[n]);
	free(receive_E2_corn8_1[n]);
	free(receive_E1_corn9_1[n]);
	free(receive_E1_corn10_1[n]);
	free(receive_E1_corn11_1[n]);
	free(receive_E1_corn12_1[n]);
#endif
	free(receive_E3_corn1_2[n]);
	free(receive_E3_corn2_2[n]);
	free(receive_E3_corn3_2[n]);
	free(receive_E3_corn4_2[n]);
#if(N3G>0)
	free(receive_E2_corn5_2[n]);
	free(receive_E2_corn6_2[n]);
	free(receive_E2_corn7_2[n]);
	free(receive_E2_corn8_2[n]);
	free(receive_E1_corn9_2[n]);
	free(receive_E1_corn10_2[n]);
	free(receive_E1_corn11_2[n]);
	free(receive_E1_corn12_2[n]);
#endif
	free(tempreceive_E3_corn1[n]);
	free(tempreceive_E3_corn2[n]);
	free(tempreceive_E3_corn3[n]);
	free(tempreceive_E3_corn4[n]);
#if(N3G>0)
	free(tempreceive_E2_corn5[n]);
	free(tempreceive_E2_corn6[n]);
	free(tempreceive_E2_corn7[n]);
	free(tempreceive_E2_corn8[n]);
	free(tempreceive_E1_corn9[n]);
	free(tempreceive_E1_corn10[n]);
	free(tempreceive_E1_corn11[n]);
	free(tempreceive_E1_corn12[n]);
#endif
	free(tempreceive_E3_corn1_1[n]);
	free(tempreceive_E3_corn2_1[n]);
	free(tempreceive_E3_corn3_1[n]);
	free(tempreceive_E3_corn4_1[n]);
#if(N3G>0)
	free(tempreceive_E2_corn5_1[n]);
	free(tempreceive_E2_corn6_1[n]);
	free(tempreceive_E2_corn7_1[n]);
	free(tempreceive_E2_corn8_1[n]);
	free(tempreceive_E1_corn9_1[n]);
	free(tempreceive_E1_corn10_1[n]);
	free(tempreceive_E1_corn11_1[n]);
	free(tempreceive_E1_corn12_1[n]);
#endif
	free(tempreceive_E3_corn1_2[n]);
	free(tempreceive_E3_corn2_2[n]);
	free(tempreceive_E3_corn3_2[n]);
	free(tempreceive_E3_corn4_2[n]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_2[n]);
	free(tempreceive_E2_corn6_2[n]);
	free(tempreceive_E2_corn7_2[n]);
	free(tempreceive_E2_corn8_2[n]);
	free(tempreceive_E1_corn9_2[n]);
	free(tempreceive_E1_corn10_2[n]);
	free(tempreceive_E1_corn11_2[n]);
	free(tempreceive_E1_corn12_2[n]);
	#endif
	free(receive_E3_corn1_12[n]);
	free(receive_E3_corn2_12[n]);
	free(receive_E3_corn3_12[n]);
	free(receive_E3_corn4_12[n]);
	#if(N3G>0)
	free(receive_E2_corn5_12[n]);
	free(receive_E2_corn6_12[n]);
	free(receive_E2_corn7_12[n]);
	free(receive_E2_corn8_12[n]);
	free(receive_E1_corn9_12[n]);
	free(receive_E1_corn10_12[n]);
	free(receive_E1_corn11_12[n]);
	free(receive_E1_corn12_12[n]);
	#endif
	free(receive_E3_corn1_22[n]);
	free(receive_E3_corn2_22[n]);
	free(receive_E3_corn3_22[n]);
	free(receive_E3_corn4_22[n]);
	#if(N3G>0)
	free(receive_E2_corn5_22[n]);
	free(receive_E2_corn6_22[n]);
	free(receive_E2_corn7_22[n]);
	free(receive_E2_corn8_22[n]);
	free(receive_E1_corn9_22[n]);
	free(receive_E1_corn10_22[n]);
	free(receive_E1_corn11_22[n]);
	free(receive_E1_corn12_22[n]);
	#endif
	#endif
	//#endif
	free(Katm[n]);
	free(array[n]);
	free(array_rdump[n]);
	free(array_gdump1[n]);
	free(array_gdump2[n]);
	free(array_diag[n]);
}


int index_3D(int n, int i, int j, int z)
{
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(N3_GPU[n] + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
}
int index_2D(int n, int i, int j, int z)
{
	#if(!NONSYMMETRIC)
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G) + ((j - N2_GPU_offset[n]) + N2G));
	#else
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(N3_GPU[n] + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
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
	int i,j,z,k,i1,j1,z1 ;
	double r, th, phi;
	struct of_geom geom ;

	/* set up boundaries, steps in coordinate grid */
	set_points(n) ;
	dV = dx[n][1] * dx[n][2] * dx[n][3];
	double X[NDIM];

	double temp = a;
	#pragma omp parallel private(X,i,j,z,k,geom, i1,j1,z1,r,th,phi,a)
	{
		DLOOPA X[j] = 0.;
		#pragma omp for collapse(2) schedule(dynamic)
		#if(!NONSYMMETRIC)
		ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]) {
		#else
		ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		#endif
			if (j<0 || j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND) a = -temp;
			else a = temp;
			
			/* zone-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND == -1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n,i, j, z, CENT, X);
			gcov_func(X, gcov[n][index_2D(n, i, j, z)][CENT]);
			gdet[n][index_2D(n, i, j, z)][CENT] = gdet_func(gcov[n][index_2D(n, i, j, z)][CENT]);
			if (j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])-1 && TRANS_BOUND == 1)gdet[n][index_2D(n, i, j, z)][CENT] *= 1.0;
			gcon_func(gcov[n][index_2D(n, i, j, z)][CENT], gcon[n][index_2D(n, i, j, z)][CENT]);
			get_geometry(n, i, j, z, CENT, &geom);
			conn_func(X, &geom, conn[n][index_2D(n, i, j, z)]);
			if ((j == -1 || j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])) && (TRANS_BOUND==1)){
				//for (i1 = 0; i1 < NDIM; i1++)for (j1 = 0; j1 < NDIM; j1++)for (z1 = 0; z1 < NDIM; z1++)conn[n][index_2D(n, i, j, z)][i1][j1][z1] = 0.;
			}

			/* corner-centered */
			/*if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n, i, j, z, FACE1, X);
			gcov_func(X, gcov[n][index_2D(n, i, j, z)][CORN]);
			gdet[n][index_2D(n, i, j, z)][CORN] = gdet_func(gcov[n][index_2D(n, i, j, z)][CORN]);
			gcon_func(gcov[n][index_2D(n, i, j, z)][CORN], gcon[n][index_2D(n, i, j, z)][CORN]);*/

			/* r-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, CORN, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, CORN, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, CORN, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, CORN, X);
			else coord(n, i, j, z, FACE1, X);
			gcov_func(X, gcov[n][index_2D(n, i, j, z)][FACE1]);
			gdet[n][index_2D(n, i, j, z)][FACE1] = gdet_func(gcov[n][index_2D(n, i, j, z)][FACE1]);
			gcon_func(gcov[n][index_2D(n, i, j, z)][FACE1], gcon[n][index_2D(n, i, j, z)][FACE1]);
			
			/* phi-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n, i, j, z, FACE3, X);
			gcov_func(X, gcov[n][index_2D(n, i, j, z)][FACE3]);
			gdet[n][index_2D(n, i, j, z)][FACE3] = gdet_func(gcov[n][index_2D(n, i, j, z)][FACE3]);
			gcon_func(gcov[n][index_2D(n, i, j, z)][FACE3], gcon[n][index_2D(n, i, j, z)][FACE3]);

			/* theta-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==1){
				//coord(n, i, 1, z, FACE2, X);
				a = 0. ;
				coord(n, i, j, z, FACE2, X);
			}
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==1){
				//coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
				coord(n, i, j, z, FACE2, X);
				a = 0.;
			}
			else coord(n, i, j, z, FACE2, X);
			gcov_func(X, gcov[n][index_2D(n, i, j, z)][FACE2]);
			gdet[n][index_2D(n, i, j, z)][FACE2] = gdet_func(gcov[n][index_2D(n, i, j, z)][FACE2]);
			gcon_func(gcov[n][index_2D(n, i, j, z)][FACE2], gcon[n][index_2D(n, i, j, z)][FACE2]);	
		}
	}

	#if(LEER)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		//Set temporary array with r, th, phi distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		coord(n, i, j, z, FACE1, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][0] = r;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][3] = r;

		coord(n, i, j, z, FACE2, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][1] = th;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][4] = th;

		coord(n, i, j, z, FACE3, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][2] = phi;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index_3D(n, i, j, z)][5] = phi;

		for (k = 0; k < 6; k++) V[n][index_3D(n, i, j, z)][k] = 0.0;
	}
	ZSLOOP3D(-N1G + N1_GPU_offset[n],-N1G + N1_GPU_offset[n], -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		V[n][index_3D(n, i, j, z)][3] = V[n][index_3D(n, i, j, z)][0] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE1][1][1]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], -N2G + N2_GPU_offset[n], -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		V[n][index_3D(n, i, j, z)][4] = V[n][index_3D(n, i, j, z)][1] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE2][2][2]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], -N3G + N3_GPU_offset[n]) {
		V[n][index_3D(n, i, j, z)][5] = V[n][index_3D(n, i, j, z)][2] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE3][3][3]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1+ N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		//Calculate distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		V[n][index_3D(n, i, j, z)][0] = V[n][index_3D(n, i - D1, j, z)][3] + 0.5*sqrt(gcov[n][index_2D(n, i - D1, j, z)][CENT][1][1]);
		V[n][index_3D(n, i, j, z)][3] = V[n][index_3D(n, i, j, z)][0] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE1][1][1]);
		
		V[n][index_3D(n, i, j, z)][1] = V[n][index_3D(n, i, j - D2, z)][4] + 0.5*sqrt(gcov[n][index_2D(n, i, j - D2, z)][CENT][2][2]);
		V[n][index_3D(n, i, j, z)][4] = V[n][index_3D(n, i, j, z)][1] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE2][2][2]);

		V[n][index_3D(n, i, j, z)][2] = V[n][index_3D(n, i, j, z - D3)][5] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z - D3)][CENT][3][3]);
		V[n][index_3D(n, i, j, z)][5] = V[n][index_3D(n, i, j, z)][2] + 0.5*sqrt(gcov[n][index_2D(n, i, j, z)][FACE3][3][3]);
	}
	#endif

	a=temp;

	#if ZIRI_DUMP
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		coord(n,i, j, z, CENT, X);
		dxdxp_func(X, dxdxp_z[n][index_3D(n ,i,j,z)]);
		//invert_matrix(dxdxp_z[n][index_3D(n ,i,j,z)], dxpdx_z[n][index_3D(n ,i,j,z)]);
	}
	#endif

	/* done! */
}