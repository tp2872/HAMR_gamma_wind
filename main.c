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
	double tdump, tdump_reduced, tlog, dump_cnt0;
	int nfailed = 0;
	int i, j, z, u, n, l;
	double r, th, phi, X[NDIM];
	clock_t begin2;
	nstep = 0;
	defcon = 1.;

	//Check input parameters
	check_input();

	/* Perform Initializations, either directly or via checkpoint */
	MPI_initialize(argc, argv);

	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_init();
	#endif
    set_AMR();

	#if (DOHELM)
	eos_init();
		#if(GPU_ENABLED || GPU_DEBUG )
		eos_init_GPU();
		#endif
	#endif

	if (!restart_read()) {
		#if(DEREFINE_POLE)
		derefine_pole();
		#endif
		for (l = 0; l < N_LEVELS_3D; l++) {
			init();
			average_grid();
			#if(N_LEVELS_3D>0)
			check_refcrit();
			#endif
		}	
		//restart_write();
		//close_rdump();
	}

	/* do initial diagnostics */
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	#endif
	diag(INIT_OUT);
	dump_cnt0 = dump_cnt;

	/*Set dumping frequency*/
	DTl = 50.0;
	DTd = 25.0;
	DTd_reduced = 50.0;
	tdump = t + DTd;
	tdump_reduced = t + DTd_reduced;
	tlog = t + DTl;
	tref = t;

	/*Start timer*/
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
		#if(RESISTIVE)
		step_ch_res();
		#else
		step_ch();
		#endif
		#endif

		/*Used for debugging*/
		#if(GPU_DEBUG)
		step_ch_debug();
		#endif

		/* deal with failed timestep, exit upon failure */
		if (failed) {
			fprintf(stderr, "Failure of some sort \n");
			break;
		}

		//Every swithchtime read out data from GPU and set boundary
		if ((nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0 && TIMER) || (t >= tref && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) || (t >= tlog && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) || (t >= tdump && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) || (t >= tdump_reduced && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0 && DUMP_SMALL)){
			end1 = get_wall_time();
			#if (GPU_ENABLED==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			bound_prim(p, 1);
			#if(!CARTESIAN)
			if (dt > 0.5) {
				fprintf(stderr, "dt too big \n");
				break;
			}
			#endif
		}

		//Refine every TREF
		if (t >= tref && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			set_timelevel(1);
			check_refcrit();
			#if (GPU_ENABLED==1)
			GPU_boundprim(1);
			#endif
			if (rank == 0) fprintf(stderr, "Refinement  succesfull! \n");
			tref += TREF;
		}

		//Put out log file and rdump file
		if (t >= tlog && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			restart_write(); //do restart dump simultaneous with log
			tlog += DTl;
		}

		/* Put out dump file*/
		if (t >= tdump && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			diag(DUMP_OUT) ;
			tdump += DTd;
		}

		/* Put out reduced dump file*/
		#if(DUMP_SMALL)
		if (t >= tdump_reduced && nstep % (DUMPFACTOR * AMR_SWITCHTIMELEVEL) == 0) {
			diag(DUMP_OUT_REDUCED);
			tdump_reduced += DTd_reduced;
		}
		#endif

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
			if (dump_cnt-dump_cnt0>50000){
				if(rank==0) fprintf(stderr, "Finishing simulation after 24 hour time period! \n");
				//restart_write();
				break;
			}
			begin1 = get_wall_time();
		}
		#endif
	}
	//cuProfilerStop();

	/* do final diagnostics */
	#if (GPU_ENABLED==1)
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	#endif
	diag(DUMP_OUT);
	diag(FINAL_OUT) ;

	/*Close GPU*/
	for (n = 0; n < n_active; n++){
		free_arrays(n_ord[n]);
		#if(GPU_ENABLED)
		GPU_finish(n_ord[n], 1);
		#endif
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
		#ifdef __APPLE__
        threadid = 0;
        nthreads = 1;
		#else
        threadid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		#endif
		if (threadid == 0 && rank == 0) {
			fprintf(stderr, "nthreads = %d\n", nthreads);
		}
	}
	//omp_set_num_threads(1);

	if (rank == 0){
		system("mkdir dumps gdumps rdumps0 rdumps1 reduced");
		#if defined(_WIN32)
		system("mkdir reduced\\gdumps");
		#else
		system("mkdir -p reduced/gdumps");
		#endif

	}
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
	#if(FRAME_TRANSFORM)
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

//Runs checks on input
void check_input() {
	
	//Select a grid that is compatible with DEREFINE_POLE
	if (DEREFINE_POLE && (NB_2 == 6 || NB_2 == 12 || NB_2 == 24 || NB_2 == 48 || NB_2 == 96)) {}
	else if(DEREFINE_POLE){
		fprintf(stderr, "Init error 1");
		exit(0);
	}

	//You can only select on version
	if (VARGAMMA + FIXEDGAMMA + CONSTANTGAMMA != 1) {
		fprintf(stderr, "Init error 2");
		exit(0);
	}

	//NB_3 has to be even in 3D
	if (NB_3 % 2 == 0 || (NB_3 * BS_3 == 1)) {}
	else {
		fprintf(stderr, "Init error 3");
		exit(0);
	}

	//Don't use block sizes this small in any case
	if ((BS_3 < 8 && NB_3 * BS_3 > 1)|| BS_2 < 8 || BS_1 < 8) {
		fprintf(stderr, "Init error 4");
		exit(0);
	}

	if (((BS_3%2 != 0) && (NB_3 * BS_3 > 1)) || BS_2 % 2 != 0 || BS_1 % 2 != 0) {
		fprintf(stderr, "Init error 5");
		exit(0);
	}
	
	//You can't run on CPU and GPU
	if (GPU_ENABLED + CPU_OPENMP > 1) {
		fprintf(stderr, "Init error 6");
		exit(0);
	}

	//Photon number evolution needs M1
	if (P_NUM && !RAD_M1) {
		fprintf(stderr, "Init error 7");
		exit(0);
	}

	//These features are not supported anymore
	if (FULL_ENTROPY || !DOKTOT) {
		fprintf(stderr, "Init error 8");
		exit(0);
	}

	//PPM not implemented in CPU version
	if (CPU_OPENMP && PPM) { 
		fprintf(stderr, "Init error 9"); 
		exit(0);
	}

	//Don't use block sizes this small on GPU
	if ((BS_3 < 16 && NB_3 * BS_3 > 1) || BS_2 < 16 || BS_1 < 16) {
		fprintf(stderr, "Init error 10");
		exit(0);
	}

	if (BS_3 / (int)pow(2, N_LEVELS_1D_INT) < 4 && N_LEVELS_1D_INT > 0) {
		if (rank == 0) fprintf(stderr, "Grid too small for number of internal derefinement levels! \n");
		//exit(0);
	}

	if (BS_2 % (int)pow(2, N_LEVELS_1D_INT) != 0 || BS_3 % (int)pow(2, N_LEVELS_1D_INT) != 0) {
		if (rank == 0) fprintf(stderr, "Grid not power of 2 of internal derefinment levels! \n");
		//exit(0);
	}

	#if(DUMP_SMALL)
	if ((BS_1 % REDUCE_FACTOR1 != 0 || BS_2 % REDUCE_FACTOR2 != 0 || BS_3 % REDUCE_FACTOR3 != 0) && DUMP_SMALL) {
		if (rank == 0) fprintf(stderr, "Grid reduction incompatible with grid size! \n");
		exit(0);
	}
	#endif
}