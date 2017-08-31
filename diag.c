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
	double divb,divbmax;
	int imax,jmax,zmax;

	/* calculate conserved quantities */
	if (call_code == INIT_OUT || call_code == LOG_OUT || call_code == FINAL_OUT) {
		divbmax = 0.;
		imax = 0;
		jmax = 0;
		zmax = 0.;
		for (n = 0; n < n_active; n++){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU_offset[n_ord[n]] + N1_GPU[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
				divb = divb_calc(n_ord[n], i, j, z);
				if (divb > divbmax && i > 1 && j > 0 && (z > 0 || N3 == 1)) {
					imax = i*pow(1 + REF_1, N_LEVELS - 1 - block[n_ord[n]][AMR_LEVEL]);
					jmax = j*pow(1 + REF_2, N_LEVELS - 1 - block[n_ord[n]][AMR_LEVEL]);
					zmax = z*pow(1 + REF_3, N_LEVELS - 1 - block[n_ord[n]][AMR_LEVEL]);
					divbmax = divb;
				}
				//if (divb > 0.00000001 && i>0) printf("divb: %f n: %d level: %d POLE: %d COORD1: %d COORD2: %d COORD3: %d  i: %d, j: %d z: %d NBR_3: %d NBR_5: %d CORN10: %d CORN10D_1: %d, CORN10D_2: %d  \n", 
				//	divb, n, block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_POLE], block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], block[n_ord[n]][AMR_COORD3], i - N1_GPU_offset[n_ord[n]], j - N2_GPU_offset[n_ord[n]], z - N3_GPU_offset[n_ord[n]],
				//	block[block[n_ord[n]][AMR_NBR3]][AMR_ACTIVE], block[block[n_ord[n]][AMR_NBR5]][AMR_ACTIVE], block[n_ord[n]][AMR_CORN10D], block[n_ord[n]][AMR_CORN10D_1], block[n_ord[n]][AMR_CORN10D_2]);
			}
		}
		#if (MPI_enable)
		MPI_Barrier(mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &divbmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &imax, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &jmax, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &zmax, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Barrier(mpi_cartcomm);
		#endif
		icurr = imax;
		jcurr = jmax;
		
		if (rank == 0){
			fprintf(stderr, "LOG      t=%g \t divbmax: %d %d %d %g\n", t, imax, jmax, zmax, divbmax);
		}
		if (divbmax > 1.){
			//fail(FAIL_GAMMA);
		}
	}

	/* gdump only at code start */
	if (call_code == INIT_OUT) {
		// make regular dump and gdump file
		#if (MPI_enable)
		MPI_Barrier(mpi_cartcomm);
		#endif
		if (rank == 0){
			fprintf(stderr, "GDUMP started \n");
		}
		gdump_new();
	}

	// dump at regular intervals 
	if (call_code == INIT_OUT ||
		call_code == DUMP_OUT ||
		call_code == FINAL_OUT) {
		// make regular dump file 
		#if (MPI_enable)
		MPI_Barrier(mpi_cartcomm);
		#endif
		if (rank == 0){
			fprintf(stderr, "DUMP%d started \n", dump_cnt);
		}
		gdump_new();
		dump_new();
		dump_cnt++;
	}
}

/** some diagnostic routines **/
void fail(int fail_type)
{
	int n;
	failed = 1 ;

	fprintf(stderr,"\n\nfail: %d %d %d\n",icurr,jcurr,fail_type) ;

	area_map(icurr,jcurr, 0, p) ;
	
	fprintf(stderr,"fail, Matthew: Doesn't work anymore correctly for AMR!\n") ;

	//diag(FINAL_OUT) ;

	/* for diagnostic purposes */
	exit(0) ;
}

/* map out region around failure point */
void area_map(int i, int j, int n, double (*restrict prim[NB])[NPR])
{
	int k ;
	fprintf(stderr,"area map\n") ;
	PLOOP {
		fprintf(stderr,"variable %d \n",k) ;
		fprintf(stderr,"i = \t %12d %12d %12d\n",i-1,i,i+1) ;
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j+1,
				prim[n_ord[n]][index(n_ord[n] ,i-1, j+1, 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i, j + 1, 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i + 1, j + 1, 0)][k]);
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j,
				prim[n_ord[n]][index(n_ord[n] ,i - 1, j, 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i, j , 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i + 1, j, 0)][k]);
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j-1,
				prim[n_ord[n]][index(n_ord[n] ,i - 1, j - 1, 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i, j - 1, 0)][k],
				prim[n_ord[n]][index(n_ord[n] ,i +1, j - 1, 0)][k]);
	}
}

double divb_calc(int n, int i, int j, int z){
	int di = (N1 > 1);
	int dj = (N2 > 1);
	int dz = (N3 > 1);
	double divb;
	
	/* Constrained transport defn */
	#if(STAGGERED)
	divb = fabs(
		#if(N1>1)
		0.25*(
		+ps[n][index(n, i + di, j, z)][1] * gdet[n][index2(n, i + di, j, z)][FACE1]
		- ps[n][index(n, i, j, z)][1] * gdet[n][index2(n, i, j, z)][FACE1]
		) / dx[n][1]
		#endif
		#if(N2>1)
		+ 0.25*(
		+ps[n][index(n, i, j + dj, z)][2] * gdet[n][index2(n, i, j + dj, z)][FACE2]
		- ps[n][index(n, i, j, z)][2] * gdet[n][index2(n, i, j, z)][FACE2]
		) / dx[n][2]
		#endif
		#if(N3>1)
		+ 0.25*(
		+ps[n][index(n, i, j, z + dz)][3] * gdet[n][index2(n, i, j, z + dz)][FACE3]
		- ps[n][index(n, i, j, z)][3] * gdet[n][index2(n, i, j, z)][FACE3]
		) / dx[n][3]
		#endif
	);
	#else
	/* Flux-ct defn */
	divb = fabs(
		#if(N1>1)
		0.25*(
		+p[n][index(n, i, j, z)][B1] * gdet[n][index2(n, i, j, z)][CENT]
		+ p[n][index(n, i, j, z - dz)][B1] * gdet[n][index2(n, i, j, z - dz)][CENT]
		+ p[n][index(n, i, j - dj, z)][B1] * gdet[n][index2(n, i, j - dj, z)][CENT]
		+ p[n][index(n, i, j - dj, z - dz)][B1] * gdet[n][index2(n, i, j - dj, z - dz)][CENT]
		- p[n][index(n, i - 1, j, z)][B1] * gdet[n][index2(n, i - 1, j, z)][CENT]
		- p[n][index(n, i - 1, j, z - dz)][B1] * gdet[n][index2(n, i - 1, j, z - dz)][CENT]
		- p[n][index(n, i - 1, j - dj, z)][B1] * gdet[n][index2(n, i - 1, j - dj, z)][CENT]
		- p[n][index(n, i - 1, j - dj, z - dz)][B1] * gdet[n][index2(n, i - 1, j - dj, z - dz)][CENT]
		) / dx[n][1]
		#endif
		#if(N2>1)
		+ 0.25*(
		+p[n][index(n, i, j, z)][B2] * gdet[n][index2(n, i, j, z)][CENT]
		+ p[n][index(n, i, j, z - dz)][B2] * gdet[n][index2(n, i, j, z - dz)][CENT]
		+ p[n][index(n, i - di, j, z)][B2] * gdet[n][index2(n, i - di, j, z)][CENT]
		+ p[n][index(n, i - di, j, z - dz)][B2] * gdet[n][index2(n, i - di, j, z - dz)][CENT]
		- p[n][index(n, i, j - 1, z)][B2] * gdet[n][index2(n, i, j - 1, z)][CENT]
		- p[n][index(n, i, j - 1, z - dz)][B2] * gdet[n][index2(n, i, j - 1, z - dz)][CENT]
		- p[n][index(n, i - di, j - 1, z)][B2] * gdet[n][index2(n, i - di, j - 1, z)][CENT]
		- p[n][index(n, i - di, j - 1, z - dz)][B2] * gdet[n][index2(n, i - di, j - 1, z - dz)][CENT]
		) / dx[n][2]
		#endif
		#if(N3>1)
		+ 0.25*(
		+p[n][index(n, i, j, z)][B3] * gdet[n][index2(n, i, j, z)][CENT]
		+ p[n][index(n, i - di, j, z)][B3] * gdet[n][index2(n, i - di, j, z)][CENT]
		+ p[n][index(n, i, j - dj, z)][B3] * gdet[n][index2(n, i, j - dj, z)][CENT]
		+ p[n][index(n, i - di, j - dj, z)][B3] * gdet[n][index2(n, i - di, j - dj, z)][CENT]
		- p[n][index(n, i, j, z - 1)][B3] * gdet[n][index2(n, i, j, z - 1)][CENT]
		- p[n][index(n, i - di, j, z - 1)][B3] * gdet[n][index2(n, i - di, j, z - 1)][CENT]
		- p[n][index(n, i, j - dj, z - 1)][B3] * gdet[n][index2(n, i, j - dj, z - 1)][CENT]
		- p[n][index(n, i - di, j - dj, z - 1)][B3] * gdet[n][index2(n, i - di, j - dj, z - 1)][CENT]
		) / dx[n][3]
		#endif
	);
	#endif
	//if (divb > 0.00000001) 
	//printf("(n, i, j, z, divb): (%d, %d, %d, %d, %f)\n", n ,i,j,z, divb);
	return divb;
}
