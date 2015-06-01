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
        Astrophysical Journal, 641, 626.

   
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

#include "decs.h"
#include "defs.h"

/*****************************************************************/
/*****************************************************************
   main():
   ------

     -- Initializes, time-steps, and concludes the simulation. 
     -- Handles timing of output routines;
     -- Main is main, what more can you say.  

-*****************************************************************/
int main(int argc,char *argv[])
{
	double tdump,timage,tlog ;
	int nfailed = 0 ;
	int i, j, u;
	double r, th, phi, X[NDIM];
	int threadid;

	/* Perform Initializations, either directly or via checkpoint */
	MPI_initialize(argc, argv);
	#pragma omp parallel shared(nthreads) private(threadid)
	{
		threadid = omp_get_thread_num();
		if (threadid == 0 && rank == 0) {
			nthreads = omp_get_num_threads();
			fprintf(stderr, "nthreads = %d\n", nthreads);
		}
	}
	//omp_set_num_threads(1);

	if (rank == 0){
		system("mkdir -p dumps images");
	}
	
	if(!restart_init()) { 
	  init() ;
	} 
	nstep = 0;

	/* do initial diagnostics */
	diag(INIT_OUT) ;
	tdump = t+DTd ;
	timage = t+DTi ;
	tlog = t+DTl ;

	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	GPU_init();
	GPU_write();
	#endif

	defcon = 1. ;
	time_spent3 = 0.0;
	begin1 = clock();
	while(t < tf) {
		/* step variables forward in time */
		nstroke = 0 ;	
		nstep++;

		/*Used for performance analysis*/
		#if(GPU_BENCHMARK)
		GPU_benchmark();
		#endif

		/*Used for running OpenCL on either GPU or CPU*/
		#if(GPU_ENABLED)
		GPU_step_ch();
		#endif
		#if(CPU_OPENMP)
		step_ch();
		#endif

		/*Used for debugging*/
		#if(GPU_DEBUG)
		step_ch_debug();
		#endif
	
		#if(GPU_FAST)
			u = -1;
			do{
				u++;
				coord(u+iboumd, 5,0, FACE1, X);
				bl_coord(X, &r, &th, &phi);
			} while (r < 100 + t);
			//printf("u: %d \n", u);
			if (u>N1-1-ibound){
				u = N1-4-ibound;
			}
			global_work_size[0] = (LOCAL_WORK_SIZE - ((u +ibound+ 8)*(N2 + 2*N2G)*(N3 + 2*N3G) - global_work_offset[0]) % LOCAL_WORK_SIZE) + ((u+ibound + 8)*(N2 + 2*N2G)*(N3 + 2*N2G) - global_work_offset[0]);
		#endif
	
		/* Handle output frequencies: */
		if(t >= tdump) {
			#if(!GPU_DEBUG && !GPU_BENCHMARK && !CPU_OPENMP)
			GPU_read();
			#endif
			diag(DUMP_OUT) ;
			tdump += DTd ;
		}
		if(t >= timage) {
			#if(!GPU_DEBUG && !GPU_BENCHMARK && !CPU_OPENMP)
			GPU_read();
			#endif
			diag(IMAGE_OUT) ;
			restart_write(); //do restart dumb simultaneous with image dump
			timage += DTi ;
		}
		if(t >= tlog) {
			clFinish(commandQueueGPU);
			end1 = clock();
			#if(!GPU_DEBUG && !GPU_BENCHMARK && !CPU_OPENMP)
			GPU_read();
			#endif
			diag(LOG_OUT) ;
			tlog += DTl ;
			if (rank == 0){
				fprintf(stderr, "Time 1: %f ", (double)(end1 - begin1) / CLOCKS_PER_SEC);
				fprintf(stderr, "Time 2: %f ", time_spent3);
				fprintf(stderr, "hslope: %f \n", hslope);
				fprintf(stderr, "nstep: %d \n", nstep);
			}
			time_spent3 = 0.0;
			begin1 = clock();
		}
		#if TIMER
		if (nstep % 2 == 0){
			end1 = clock();
			if (rank == 0){
				fprintf(stderr, "Time 1: %f ", (double)(end1 - begin1) / CLOCKS_PER_SEC);
				fprintf(stderr, "Time 2: %f ", time_spent3);
				fprintf(stderr, "hslope: %f \n", hslope);
				fprintf(stderr, "nstep: %d \n", nstep);
			}
			time_spent3 = 0.0;
			begin1 = clock();
			break;
		}
		#endif

		/* deal with failed timestep, though we usually exit upon failure */
		if(failed) {
			restart_init() ;
			failed = 0 ;
			nfailed = nstep ;
			defcon = 0.3 ;
		}
		if(nstep > nfailed + DTr*4.*(1 + 1./defcon)) defcon = 1. ;
	}
	fprintf(stderr,"ns,ts: %d %d\n",nstep,nstep*N1*N2) ;

	/* do final diagnostics */
	diag(FINAL_OUT) ;

	/*Close GPU*/
	GPU_finish();
	return(0) ;
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
void set_arrays()
{
	int i, j, z, k;
	pbound = (double(*)[NPR][N_POINTS])malloc((N2 + 2 * N2G)*(N3_MPI + 2 * N3G) * sizeof(double[NPR][N_POINTS]));
	p = (double(*)[NPR])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(double[NPR]));
	psave = (double(*)[NPR])malloc((N1 + 2*N1G)*(N2 + 2*N2G)*sizeof(double[NPR]));
	fsave = (double(*)[NFAIL])malloc((N1 + 2*N1G)*(N2 + 2*N2G)* sizeof(double[NFAIL]));
	dq = (double(*)[NPR])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(double[NPR]));
	#if(GPU_BENCHMARK || GPU_DEBUG || CPU_OPENMP)
	F1 = (double(*)[NPR])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(double[NPR]));
	F2 = (double(*)[NPR])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(double[NPR]));
	F3 = (double(*)[NPR])malloc((N1_MPI + 2 * N1G)*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G)* sizeof(double[NPR]));
	ph = (double(*)[NPR])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(double[NPR]));
	#endif
	pflag = (int(*))malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(int));
	failimage = (int(*)[NFAIL])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)* sizeof(int[NFAIL]));
	conn = (double(*)[NDIM][NDIM][NDIM])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)* sizeof(double[NDIM][NDIM][NDIM]));
	gcov = (double(*)[NPG][NDIM][NDIM])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G) * sizeof(double[NPG][NDIM][NDIM]));
	gcon = (double(*)[NPG][NDIM][NDIM])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)* sizeof(double[NPG][NDIM][NDIM]));
	gdet = (double(*)[NPG])malloc((N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)* sizeof(double[NPG]));
	send = (double *)malloc(2 * NPR*(N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G) * sizeof(double));
	receive1 = (FTYPE2 *)malloc(2 * NPR*(N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	receive2 = (FTYPE2 *)malloc(2 * NPR*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	receive3 = (FTYPE2 *)malloc(2 * NPR*(N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	receive4 = (FTYPE2 *)malloc(2 * NPR*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	send1 = (FTYPE2 *)malloc(2 * NPR*(N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	send2 = (FTYPE2 *)malloc(2 * NPR*(N2_MPI + 2 * N2G) *(N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	send3 = (FTYPE2 *)malloc(2 * NPR*(N1_MPI + 2 * N1G) *(N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	send4 = (FTYPE2 *)malloc(2 * NPR*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	receive = (double *)malloc(2 * NPR*(N1_MPI + 2*N1G)*(N2_MPI + 2*N2G)*(N3_MPI + 2*N3G)  * sizeof(double));
	cornreceive1 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornreceive2 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornreceive3 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornreceive4 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornsend1 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornsend2 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornsend3 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G)* sizeof(FTYPE2));
	cornsend4 = (FTYPE2 *)malloc(NPR * 4 * (N3_MPI + 2 * N3G) * sizeof(FTYPE2));
	fimage = (double(**))malloc(NIMG * sizeof(double *));
	for (i = 0; i < NIMG; i++){
		fimage[i] = (double(*))malloc(NIMG*N1*N2 * sizeof(double));
	}

	/* everything must be initialized to zero */
	ZSLOOP3D(-N1G + N1_MPI_offset, N1_MPI_offset + N1_MPI - 1 + N1G , -N2G + N2_MPI_offset, N2_MPI_offset + N2_MPI - 1 + N2G, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI - 1 + N3G) {
		PLOOP {
			p[index(i,j,z)][k]   = 0. ;
			psave[index3(i,j)][k] = 0.;
			dq[index(i,j,z)][k] = 0.;
			#if(GPU_BENCHMARK || GPU_DEBUG || CPU_OPENMP)
			ph[index(i,j,z)][k] = 0.;
			F1[index(i,j,z)][k] = 0.;
			F2[index(i,j,z)][k] = 0.;
			F3[index(i, j, z)][k] = 0.;
			#endif
		}
		for (k = 0; k < NFAIL; k++){
			failimage[index(i,j,z)][k] = 0;
		}
		pflag[index(i,j,z)] = 1;
	}
}

int index(int i, int j, int z)
{
	return(((i - N1_MPI_offset) + N1G)*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) + ((j - N2_MPI_offset) + N2G)*(N3_MPI + 2 * N3G) + ((z - N3_MPI_offset) + N3G));
}
int index2(int i, int j)
{
	return(((i - N1_MPI_offset) + N1G)*(N2_MPI + 2 * N2G) + ((j - N2_MPI_offset) + N2G));
}
int index3(int i, int j)
{
	return((i + N1G)*(N2 + 2 * N2G) + (j + N2G));
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
void set_grid()
{
	int i,j,z,k ;
	double X[NDIM] ;
	struct of_geom geom ;

	/* set up boundaries, steps in coordinate grid */
	set_points() ;
	dV = dx[1]*dx[2] *dx[3];
	DLOOPA X[j] = 0. ;
	z = 0;
	ZSLOOP(-N1G + N1_MPI_offset, N1_MPI + N1_MPI_offset - 1 + N1G, -N2G + N2_MPI_offset, N2_MPI_offset + N2_MPI - 1 + N2G) {
		/* zone-centered */
		coord(i,j,z,CENT,X) ;
		gcov_func(X, gcov[index2(i, j)][CENT]);
		gdet[index2(i, j)][CENT] = gdet_func(gcov[index2(i, j)][CENT]);
		gcon_func(gcov[index2(i, j)][CENT], gcon[index2(i, j)][CENT]);
		get_geometry(i,j,CENT,&geom) ;
		conn_func(X, &geom, conn[index2(i, j)]);

		/* corner-centered */
		coord(i,j,z,CORN,X) ;
		gcov_func(X, gcov[index2(i, j)][CORN]);
		gdet[index2(i, j)][CORN] = gdet_func(gcov[index2(i, j)][CORN]);
		gcon_func(gcov[index2(i, j)][CORN], gcon[index2(i, j)][CORN]);

		/* r-face-centered */
		coord(i,j,z,FACE1,X) ;
		gcov_func(X, gcov[index2(i, j)][FACE1]);
		gdet[index2(i, j)][FACE1] = gdet_func(gcov[index2(i, j)][FACE1]);
		gcon_func(gcov[index2(i, j)][FACE1], gcon[index2(i, j)][FACE1]);

		/* theta-face-centered */
		coord(i,j,z,FACE2,X) ;
		gcov_func(X, gcov[index2(i, j)][FACE2]);
		gdet[index2(i, j)][FACE2] = gdet_func(gcov[index2(i, j)][FACE2]);
		gcon_func(gcov[index2(i, j)][FACE2], gcon[index2(i, j)][FACE2]);

		/* phi-face-centered */
		coord(i, j, z, FACE3, X);
		gcov_func(X, gcov[index2(i, j)][FACE3]);
		gdet[index2(i, j)][FACE3] = gdet_func(gcov[index2(i, j)][FACE3]);
		gcon_func(gcov[index2(i, j)][FACE3], gcon[index2(i, j)][FACE3]);
	}
	/* done! */
}


/*This function initialises the MPI structure. It divides the grid(N1, N2, N3) among the MPI processes.
The host node is node 0 by default.*/
void MPI_initialize(int argc, char *argv[])
{
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int i, len;
	
	/*Get basic initialisation*/
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(hostname, &len);

	/*Handy for debugging, you don't have to change this value everytime if you want to test MPI vs non-MPI*/
	if (numtasks != 1){
		n_rows = MPI_rows;
		n_columns = MPI_columns;
		n_stacks = MPI_stacks;
	}
	else{
		n_rows = 1;
		n_columns = 1;
		n_stacks = 1;
	}

	/*Get the offset in the grid for all MPI processes.*/
	aN1_MPI_offset = (int *)malloc(numtasks*sizeof(int));
	aN2_MPI_offset = (int *)malloc(numtasks * sizeof(int));
	aN3_MPI_offset = (int *)malloc(numtasks * sizeof(int));
	aN1_MPI = (int *)malloc(numtasks * sizeof(int));
	aN2_MPI = (int *)malloc(numtasks * sizeof(int));
	aN3_MPI = (int *)malloc(numtasks * sizeof(int));
	for (i = 0; i < numtasks; i++){
		n3_MPI = (i % (n_rows*n_stacks)) % n_stacks;
		n2_MPI = ((i - n3_MPI) % (n_rows*n_stacks)) / n_stacks;
		n1_MPI = (i - (n2_MPI*n_stacks + n3_MPI)) / (n_rows*n_stacks);

		if (n1_MPI < n_columns - 1){
			aN1_MPI_offset[i] = (int)floor((double)(N1) / (double)n_columns)*n1_MPI;
			aN1_MPI[i] = (int)floor((double)(N1) / (double)n_columns);
		}
		else{
			aN1_MPI_offset[i] = (int)floor((double)(N1) / (double)n_columns)*n1_MPI;
			aN1_MPI[i] = N1 - aN1_MPI_offset[i];
		}

		if (n2_MPI < n_rows - 1){
			aN2_MPI_offset[i] = (int)floor((double)(N2) / (double)n_rows)*n2_MPI;
			aN2_MPI[i] = (int)floor((double)(N2) / (double)n_rows);
		}
		else{
			aN2_MPI_offset[i] = (int)floor((double)(N2) / (double)n_rows)*n2_MPI;
			aN2_MPI[i] = N2 - aN2_MPI_offset[i];
		}
		if (n3_MPI < n_stacks - 1){
			aN3_MPI_offset[i] = (int)floor((double)(N3) / (double)n_stacks)*n3_MPI;
			aN3_MPI[i] = (int)floor((double)(N3) / (double)n_stacks);
		}
		else{
			aN3_MPI_offset[i] = (int)floor((double)(N3) / (double)n_stacks)*n3_MPI;
			aN3_MPI[i] = N3- aN3_MPI_offset[i];
		}
	}
	N1_MPI_offset = aN1_MPI_offset[rank];
	N1_MPI = aN1_MPI[rank];
	N2_MPI_offset = aN2_MPI_offset[rank];
	N2_MPI = aN2_MPI[rank];
	N3_MPI_offset = aN3_MPI_offset[rank];
	N3_MPI = aN3_MPI[rank];

	/*Get the number of MPI cells in r, theta and phi directions*/
	n3_MPI = (rank % (n_rows*n_stacks))%n_stacks;
	n2_MPI = ((rank - n3_MPI) % (n_rows*n_stacks))/n_stacks;
	n1_MPI = (rank - (n2_MPI*n_stacks+n3_MPI)) / (n_rows*n_stacks);

	/*Give notice if programmer makes error in selecting the MPI_rows and MPI_column values in dec.h*/
	if (rank == 0 && n_rows*n_columns != numtasks){
		printf("Error: Number of tasks is not equal to rows*columns! \n");
	}

	/*Give basic diagnostics*/
	if (rank == 0){
		printf("Number of MPI tasks: %d \nRunning on: %s\n", numtasks, hostname);
		printf("MPI geometry(rows, columns, stacks) : (%d, %d, %d)\n", n_rows, n_columns, n_stacks);
	}
}

