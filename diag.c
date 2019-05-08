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
#include "decs_MPI.h"

/* all diagnostics subroutine */
void diag(int call_code);
void diag(int call_code)
{
	int i,j,z,k,n ;
	double divb,divbmax, divbmax_local;
	int imax,jmax,zmax, nmax;
	for (n = 0; n < n_active; n++) B_send1(ps, Bufferps_1, n_ord[n]);
	for (n = 0; n < n_active; n++) B_rec1(ps, Bufferps_1, n_ord[n]);	
	prolong_grid();

	/* calculate conserved quantities */
	if (call_code == INIT_OUT || call_code == LOG_OUT || call_code == FINAL_OUT) {
		divbmax = 0.;
		imax = 0;
		jmax = 0;
		zmax = 0.;
		nmax = 0;
		for (n = 0; n < n_active; n++){
			#pragma omp parallel for schedule(static,(BS_1)*(BS_2)*(BS_3)/nthreads) private(divb,i,j,z)
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU_offset[n_ord[n]] + BS_1 - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
				divb = divb_calc(n_ord[n], i, j, z);
				#pragma omp critical
				if (divb > divbmax && i > 1 && j > 0 && (z > 0 || N3 == 1)) {
					imax = i;
					jmax = j;
					zmax = z;
					nmax = n_ord[n];
					divbmax = divb;
				}

				//#pragma omp critical
				//if (divb > 0.0000001 && numtasks<100){
					//fprintf(stderr, "n: %d divb:  (%d %d)x(%d %d %d)x(%d %d %d)x(%d %d %d) %f \n", n_ord[n], block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_LEVEL1], block[n_ord[n]][AMR_LEVEL2], block[n_ord[n]][AMR_LEVEL3], block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], block[n_ord[n]][AMR_COORD3], i, j, z, divb);
				//}
			}
		}
	
		#if (MPI_enable)
		divbmax_local = divbmax;
		MPI_Allreduce(MPI_IN_PLACE, &divbmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
		//for(k=0;k<NFAIL;k++)MPI_Allreduce(MPI_IN_PLACE, &failimage_counter[k], 1, MPI_INT, MPI_SUM, mpi_cartcomm);
		#endif
		
		if (divbmax==divbmax_local){
			fprintf(stderr, "LOG      t=%g \t divbmax: (%d %d %d)x(%d %d %d)x(%d %d %d) %g \n", t, block[nmax][AMR_LEVEL1], block[nmax][AMR_LEVEL2], block[nmax][AMR_LEVEL3], block[nmax][AMR_COORD1], block[nmax][AMR_COORD2], block[nmax][AMR_COORD3], imax - N1_GPU_offset[nmax], jmax - N2_GPU_offset[nmax], zmax - N3_GPU_offset[nmax], divbmax);
			//fprintf(stderr, " f1: %d f2: %d f3: %d f4 %d \n", failimage_counter[0], failimage_counter[1], failimage_counter[2], failimage_counter[3]);
		}
	}

	if (call_code == FINAL_OUT) {
		//First close dump files in progress
		close_dump();
		close_rdump();
		close_gdump();
		#if(DUMP_SMALL)
		close_dump_reduced();
		close_gdump_reduced();
		#endif
	}

	// dump at regular intervals 
	if (call_code == DUMP_OUT_REDUCED) {
		// make regular dump file 
		if (rank == 0) {
			fprintf(stderr, "GDUMP_reduced started \n");
			fprintf(stderr, "DUMP%d_reduced started \n", dump_cnt_reduced);
		}
		gdump_new_reduced();
		dump_new_reduced();
	}

	// dump at regular intervals 
	if (call_code == INIT_OUT || call_code == DUMP_OUT) {
		// make regular dump file 
		if (rank == 0){
			fprintf(stderr, "GDUMP started \n");
			fprintf(stderr, "DUMP%d started \n", dump_cnt);
		}
		gdump_new();
		dump_new();
		#if(DUMP_SMALL)
		if(nstep==0){
			if (rank == 0) {
				fprintf(stderr, "GDUMP_reduced started \n");
				fprintf(stderr, "DUMP%d_reduced started \n", dump_cnt_reduced);
			}
			gdump_new_reduced();
			dump_new_reduced();
		}
		#endif
	}
}

/** some diagnostic routines **/
void fail(int fail_type)
{
	int n;
	failed = 1 ;

	//fprintf(stderr,"\n\nfail: %d %d %d\n",icurr,jcurr,fail_type) ;

	//area_map(icurr,jcurr, 0, p) ;
	
	fprintf(stderr,"Failed, error number %d!\n", fail_type) ;

	//diag(FINAL_OUT) ;

	/* for diagnostic purposes */
	exit(0) ;
}

/* map out region around failure point */
void area_map(int i, int j, int n, double (*restrict prim[NB_LOCAL])[NPR])
{

}

double divb_calc(int n, int i, int j, int z){
	int di = (N1 > 1);
	int dj = (N2 > 1);
	int dz = (N3 > 1);
	double divb=0.0;
	int zsize = 1, zoffset = 0, zlevel = 0, u;

	#if(N_LEVELS_1D_INT>10 && D3>0)
	if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && j < N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n]) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if ((block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) && j >= N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = round(pow(2.0, (double)zlevel));
	zoffset = (z - N3_GPU_offset[n]) % zsize;
	#endif

	/* Constrained transport defn */
	#if(STAGGERED)
	#if(N1>1)
	for (u = 0; u < zsize; u++){
		divb += 0.25*(ps[nl[n]][index_3D(n, i + di, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i + di, j, z - zoffset + u)][FACE1] - ps[nl[n]][index_3D(n, i, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE1]) / ((double)(zsize)*dx[nl[n]][1]);
	}
	#endif
	#if(N2>1)
	for (u = 0; u < zsize; u++){
		divb += 0.25*(ps[nl[n]][index_3D(n, i, j + dj, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j + dj, z - zoffset + u)][FACE2] - ps[nl[n]][index_3D(n, i, j, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE2]) / ((double)(zsize)*dx[nl[n]][2]);
	}
	#endif
	#if(N3>1)
	divb += 0.25*(ps[nl[n]][index_3D(n, i, j, z - zoffset + dz * zsize)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + dz * zsize)][FACE3] - ps[nl[n]][index_3D(n, i, j, z - zoffset)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset)][FACE3]) / ((double)(zsize)*dx[nl[n]][3]);
	#endif
	divb = fabs(divb);
	#else
	/* Flux-ct defn */
	divb = fabs(
		#if(N1>1)
		0.25*(
		+p[nl[n]][index_3D(n, i, j, z)][B1] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
		+ p[nl[n]][index_3D(n, i, j, z - dz)][B1] * gdet[nl[n]][index_2D(n, i, j, z - dz)][CENT]
		+ p[nl[n]][index_3D(n, i, j - dj, z)][B1] * gdet[nl[n]][index_2D(n, i, j - dj, z)][CENT]
		+ p[nl[n]][index_3D(n, i, j - dj, z - dz)][B1] * gdet[nl[n]][index_2D(n, i, j - dj, z - dz)][CENT]
		- p[nl[n]][index_3D(n, i - 1, j, z)][B1] * gdet[nl[n]][index_2D(n, i - 1, j, z)][CENT]
		- p[nl[n]][index_3D(n, i - 1, j, z - dz)][B1] * gdet[nl[n]][index_2D(n, i - 1, j, z - dz)][CENT]
		- p[nl[n]][index_3D(n, i - 1, j - dj, z)][B1] * gdet[nl[n]][index_2D(n, i - 1, j - dj, z)][CENT]
		- p[nl[n]][index_3D(n, i - 1, j - dj, z - dz)][B1] * gdet[nl[n]][index_2D(n, i - 1, j - dj, z - dz)][CENT]
		) / dx[nl[n]][1]
		#endif
		#if(N2>1)
		+ 0.25*(
		+p[nl[n]][index_3D(n, i, j, z)][B2] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
		+ p[nl[n]][index_3D(n, i, j, z - dz)][B2] * gdet[nl[n]][index_2D(n, i, j, z - dz)][CENT]
		+ p[nl[n]][index_3D(n, i - di, j, z)][B2] * gdet[nl[n]][index_2D(n, i - di, j, z)][CENT]
		+ p[nl[n]][index_3D(n, i - di, j, z - dz)][B2] * gdet[nl[n]][index_2D(n, i - di, j, z - dz)][CENT]
		- p[nl[n]][index_3D(n, i, j - 1, z)][B2] * gdet[nl[n]][index_2D(n, i, j - 1, z)][CENT]
		- p[nl[n]][index_3D(n, i, j - 1, z - dz)][B2] * gdet[nl[n]][index_2D(n, i, j - 1, z - dz)][CENT]
		- p[nl[n]][index_3D(n, i - di, j - 1, z)][B2] * gdet[nl[n]][index_2D(n, i - di, j - 1, z)][CENT]
		- p[nl[n]][index_3D(n, i - di, j - 1, z - dz)][B2] * gdet[nl[n]][index_2D(n, i - di, j - 1, z - dz)][CENT]
		) / dx[nl[n]][2]
		#endif
		#if(N3>1)
		+ 0.25*(
		+p[nl[n]][index_3D(n, i, j, z)][B3] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
		+ p[nl[n]][index_3D(n, i - di, j, z)][B3] * gdet[nl[n]][index_2D(n, i - di, j, z)][CENT]
		+ p[nl[n]][index_3D(n, i, j - dj, z)][B3] * gdet[nl[n]][index_2D(n, i, j - dj, z)][CENT]
		+ p[nl[n]][index_3D(n, i - di, j - dj, z)][B3] * gdet[nl[n]][index_2D(n, i - di, j - dj, z)][CENT]
		- p[nl[n]][index_3D(n, i, j, z - 1)][B3] * gdet[nl[n]][index_2D(n, i, j, z - 1)][CENT]
		- p[nl[n]][index_3D(n, i - di, j, z - 1)][B3] * gdet[nl[n]][index_2D(n, i - di, j, z - 1)][CENT]
		- p[nl[n]][index_3D(n, i, j - dj, z - 1)][B3] * gdet[nl[n]][index_2D(n, i, j - dj, z - 1)][CENT]
		- p[nl[n]][index_3D(n, i - di, j - dj, z - 1)][B3] * gdet[nl[n]][index_2D(n, i - di, j - dj, z - 1)][CENT]
		) / dx[nl[n]][3]
		#endif
	);
	#endif
	return divb;
}