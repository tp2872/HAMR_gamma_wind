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

void FMSS_write(FILE *fp){
	int i, j, z;
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	double i_double, j_double, z_double;
	printf("test");
	int n = 0;
	ZSLOOP3D(0, N1_GPU[n_ord[n]] - 1, 0, N2_GPU[n_ord[n]] - 1, 0, N3_GPU[n_ord[n]] - 1) {
		i_double = (double)i;
		j_double = (double)j;
		z_double = (double)z;
		fwrite(&i_double, double_size, 1, fp);
		fwrite(&j_double, double_size, 1, fp);
		fwrite(&z_double, double_size, 1, fp);
		fwrite(&(connected[index2(n_ord[n] ,i, j, 0)]), double_size, 1, fp);
	}
}

void gdump_grid(FILE *fp)
{
	int n, k;
	int int_size = sizeof(int);
	if (rank == 0){
		int NB_print = NB;
		fwrite(&NB_print, int_size, 1, fp);
		for (n = 0; n <= n_max; n++){
			for (k = 0; k < 36; k++){ //SASMARK: why is 36 hard-coded?
				fwrite(&(block[n][k]), int_size, 1, fp);
			}
		}
	}
}

void dump(FILE *fp)
{
	int i,j,z,k,n ;
	int di, dj, dz;
	double i_double, j_double, z_double;
	double divb ;
	double X[NDIM] ;
	double r,th,phi,vmin,vmax ;
	struct of_geom geom ;
	struct of_state q ;
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	char newline[] = { '\n'};
	/***************************************************************
	  Write header information : 
	***************************************************************/
	if (rank == 0){
		int N1_print = BS_1;
		int N2_print = BS_2;
		int N3_print = BS_3;
		fwrite(&t, double_size, 1, fp);
		fwrite(&N1_print, int_size, 1, fp);
		fwrite(&N2_print, int_size, 1, fp);
		fwrite(&N3_print, int_size, 1, fp);
		fwrite(&n_active, int_size, 1, fp);
		fwrite(&n_rows, int_size, 1, fp);
		fwrite(&n_columns, int_size, 1, fp);
		fwrite(&n_stacks, int_size, 1, fp);
		fwrite(&startx[1], double_size, 1, fp);
		fwrite(&startx[2], double_size, 1, fp);
		fwrite(&startx[3], double_size, 1, fp);
		fwrite(&dx[0][1], double_size, 1, fp);
		fwrite(&dx[0][2], double_size, 1, fp);
		fwrite(&dx[0][3], double_size, 1, fp);
		fwrite(&tf, double_size, 1, fp);
		fwrite(&nstep, int_size, 1, fp);
		fwrite(&a, double_size, 1, fp);
		fwrite(&gam, double_size, 1, fp);
		fwrite(&cour, double_size, 1, fp);
		fwrite(&DTd, double_size, 1, fp);
		fwrite(&DTl, double_size, 1, fp);
		fwrite(&DTi, double_size, 1, fp);
		fwrite(&DTr, int_size, 1, fp);
		fwrite(&dump_cnt, int_size, 1, fp);
		fwrite(&image_cnt, int_size, 1, fp);
		fwrite(&rdump_cnt, int_size, 1, fp);
		fwrite(&dt, double_size, 1, fp);
		fwrite(&lim, int_size, 1, fp);
		fwrite(&failed, int_size, 1, fp);
		fwrite(&Rin, double_size, 1, fp);
		fwrite(&Rout, double_size, 1, fp);
		fwrite(&hslope, double_size, 1, fp);
		fwrite(&R0, double_size, 1, fp);
	}
	/***************************************************************
	  Write header information : 
	***************************************************************/
	di = (N1>1);
	dj = (N2>1);
	dz = (N3>1);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU_offset[n_ord[n]] + N1_GPU[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			i_double = (double)i;//*pow(2., (N_LEVELS - (block[n_ord[n]][AMR_LEVEL] + 1))*REF_1);
			j_double = (double)j;//*pow(2., (N_LEVELS - (block[n_ord[n]][AMR_LEVEL] + 1))*REF_2);
			z_double = (double)z;//*pow(2., (N_LEVELS - (block[n_ord[n]][AMR_LEVEL] + 1))*REF_3);
			fwrite(&i_double, double_size, 1, fp);
			fwrite(&j_double, double_size, 1, fp);
			fwrite(&z_double, double_size, 1, fp);
			fwrite(&X[1], double_size, 1, fp);

			fwrite(&X[2], double_size, 1, fp);
			fwrite(&X[3], double_size, 1, fp);
			fwrite(&r, double_size, 1, fp);
			fwrite(&th, double_size, 1, fp);
			fwrite(&phi, double_size, 1, fp);
			for (k = 0; k < 8; k++) fwrite(&(p[n_ord[n]][index(n_ord[n] ,i, j, z)][k]), double_size, 1, fp);

			/* divb flux-ct defn; corner-centered.  Useonly interior corners */
			if (i > 0 && j > 0 && i < N1 && j < N2) {
				divb = fabs(
					#if(N1>1)
					0.25*(
					+p[n_ord[n]][index(n_ord[n] ,i, j, z)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i, j, z - dz)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z - dz)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i, j - dj, z)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - dj, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i, j - dj, z - dz)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - dj, z - dz)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - 1, j, z)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i - 1, j, z)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - 1, j, z - dz)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i - 1, j, z - dz)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - 1, j - dj, z)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i - 1, j - dj, z)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - 1, j - dj, z - dz)][B1] * gdet[n_ord[n]][index2(n_ord[n] ,i - 1, j - dj, z - dz)][CENT]
					) / dx[n_ord[n]][1]
					#endif
					#if(N2>1)
					+ 0.25*(
					+p[n_ord[n]][index(n_ord[n] ,i, j, z)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i, j, z - dz)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z - dz)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i - di, j, z)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i - di, j, z - dz)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j, z - dz)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i, j - 1, z)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - 1, z)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i, j - 1, z - dz)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - 1, z - dz)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - di, j - 1, z)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j - 1, z)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - di, j - 1, z - dz)][B2] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j - 1, z - dz)][CENT]
					) / dx[n_ord[n]][2]
					#endif
					#if(N3>1)
					+ 0.25*(
					+p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i - di, j, z)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i, j - dj, z)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - dj, z)][CENT]
					+ p[n_ord[n]][index(n_ord[n] ,i - di, j - dj, z)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j - dj, z)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i, j, z - 1)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i, j, z - 1)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - di, j, z - 1)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j, z - 1)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i, j - dj, z - 1)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i, j - dj, z - 1)][CENT]
					- p[n_ord[n]][index(n_ord[n] ,i - di, j - dj, z - 1)][B3] * gdet[n_ord[n]][index2(n_ord[n] ,i - di, j - dj, z - 1)][CENT]
					) / dx[n_ord[n]][3]
					#endif
					);
			}
			else divb = 0.;

			fwrite(&divb, double_size, 1, fp);

			if (!failed) {
				get_geometry(n_ord[n], i, j, z, CENT, &geom);
				get_state(p[n_ord[n]][index(n_ord[n] ,i, j, z)], &geom, &q);

				for (k = 0; k < NDIM; k++) fwrite(&(q.ucon[k]), double_size, 1, fp);
				for (k = 0; k < NDIM; k++) fwrite(&(q.ucov[k]), double_size, 1, fp);
				for (k = 0; k < NDIM; k++) fwrite(&(q.bcon[k]), double_size, 1, fp);
				for (k = 0; k < NDIM; k++) fwrite(&(q.bcov[k]), double_size, 1, fp);

				vchar(p[n_ord[n]][index(n_ord[n] ,i, j, z)], &q, &geom, 1, &vmax, &vmin, i, j, z);
				fwrite(&vmin, double_size, 1, fp);
				fwrite(&vmax, double_size, 1, fp);

				vchar(p[n_ord[n]][index(n_ord[n] ,i, j, z)], &q, &geom, 2, &vmax, &vmin, i, j, z);
				double v0 = (double)(failimage[n_ord[n]][index(n_ord[n], i, j, z)][0]);
				double v1 = (double)(failimage[n_ord[n]][index(n_ord[n], i, j, z)][1]);
				fwrite(&v0, double_size, 1, fp);
				fwrite(&v1, double_size, 1, fp);
				//fwrite(&vmin, double_size, 1, fp);
				//fwrite(&vmax, double_size, 1, fp);


				fwrite(&(geom.g), double_size, 1, fp);
			}
		}
	}
}

void dump_new(void){
	int n, u;
	char filename[100], dirpath[100];

	if (rank == 0){
		FILE *fparam;
		sprintf(dirpath, "mkdir dumps%d", dump_cnt);
		system(dirpath);
		sprintf(filename, "dumps%d/parameters", dump_cnt);
		fparam = fopen(filename, "wb");
		dump_params(fparam);
		fclose(fparam);
	}

	if (rank == 0){
		FILE *grid;
		sprintf(filename, "dumps%d/grid", dump_cnt);
		grid = fopen(filename, "wb");
		gdump_grid(grid);
		fclose(grid);
	}
	
	int u_stride=200;
	int u_max=(n_active_total-n_active_total%u_stride)/u_stride;
	if(n_active_total%u_stride!=0) u_max++;
	
	first_dump = 1;
	
	for(u=0; u<u_max; u++){
		sprintf(filename, "dumps%d/new_dump%d", dump_cnt, u);
		MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL, &fdump[u]);
	}
	for (n = 0; n < n_active_total; n++){
		for(u=0; u<u_max; u++)if(n>=u*u_stride && n<(u+1)*u_stride){
			if (block[n_ord_total[n]][AMR_NODE]==rank){
				MPI_File_seek(fdump[u], (n-u*u_stride) * 9 * BS_1*BS_2*BS_3*sizeof(float), MPI_SEEK_SET);
				dump_block(&fdump[u], n_ord_total[n]);
			}
		}
	}

	if (dump_cnt%1==0){
		for(u=0; u<u_max; u++){
			sprintf(filename, "dumps%d/new_dumpdiag%d", dump_cnt, u);
			MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL, &fdumpdiag[u]);
		}
		for (n = 0; n < n_active_total; n++){
			for(u=0; u<u_max; u++)if(n>=u*u_stride && n<(u+1)*u_stride){
				if (block[n_ord_total[n]][AMR_NODE] == rank){
					MPI_File_seek(fdumpdiag[u], (n-u*u_stride) * 4 * BS_1*BS_2*BS_3*sizeof(float), MPI_SEEK_SET);
					dump_blockdiag(&fdumpdiag[u], n_ord_total[n]);
				}
			}
		}
	}
}

void gdump_new(void){
	int n;
	char filename[100];

	FILE *gdump;
	FILE *grid;
	if (rank == 0){
		sprintf(filename, "gdumps/grid");
		grid = fopen(filename, "wb");
		gdump_grid(grid);
		fclose(grid);
	}
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][GDUMP_WRITTEN] != 1){
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				sprintf(filename, "gdumps/gdump%d", n_ord_total[n]);
				gdump = fopen(filename, "wb");
				gdump_block(gdump, n_ord_total[n]);
				fclose(gdump);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN] = 1;
		}
	}
}

void dump_params(FILE *fp)
{
	int u;
	int int_size = sizeof(int);
	int double_size = sizeof(double);

	if (rank == 0){
		int BS1_print = BS_1;
		int BS2_print = BS_2;
		int BS3_print = BS_3;
		int NB_print = NB;
		int NB1_print = NB_1;
		int NB2_print = NB_2;
		int NB3_print = NB_3;
		int stag = STAGGERED;
		int B = BRAVO;
		int T = TANGO;
		int C = CHARLIE;
		int D = DELTA;
		int r1 = REF_1;
		int r2 = REF_2;
		int r3 = REF_3;
		int nl = N_LEVELS;
		int rx = RADEXP;
		int rt = RTRANS;
		int rb = RB;
		int docyl = DOCYLINDRIFYCOORDS;
		int dk = DOKTOT;

		//Print out essential stuff for restart
		fwrite(&t, double_size, 1, fp);
		fwrite(&n_active, int_size, 1, fp);
		fwrite(&n_active_total, int_size, 1, fp);
		fwrite(&nstep, int_size, 1, fp);
		fwrite(&DTd, double_size, 1, fp);
		fwrite(&DTl, double_size, 1, fp);
		fwrite(&DTr, double_size, 1, fp);
		fwrite(&dump_cnt, int_size, 1, fp);
		fwrite(&rdump_cnt, int_size, 1, fp);
		fwrite(&dt, double_size, 1, fp);
		fwrite(&failed, int_size, 1, fp);

		//Print out stuff that should be checked later
		fwrite(&BS1_print, int_size, 1, fp);
		fwrite(&BS2_print, int_size, 1, fp);
		fwrite(&BS3_print, int_size, 1, fp);
		fwrite(&NB_print, int_size, 1, fp);
		fwrite(&NB1_print, int_size, 1, fp);
		fwrite(&NB2_print, int_size, 1, fp);
		fwrite(&NB3_print, int_size, 1, fp);
		fwrite(&startx[1], double_size, 1, fp);
		fwrite(&startx[2], double_size, 1, fp);
		fwrite(&startx[3], double_size, 1, fp);
		fwrite(&dx[0][1], double_size, 1, fp);
		fwrite(&dx[0][2], double_size, 1, fp);
		fwrite(&dx[0][3], double_size, 1, fp);
		
		fwrite(&tf, double_size, 1, fp);
		fwrite(&a, double_size, 1, fp);
		fwrite(&gam, double_size, 1, fp);
		fwrite(&cour, double_size, 1, fp);
		fwrite(&Rin, double_size, 1, fp);
		fwrite(&Rout, double_size, 1, fp);
		fwrite(&R0, double_size, 1, fp);		
		fwrite(&fractheta, double_size, 1, fp);
		fwrite(&lim, int_size, 1, fp);
		fwrite(&stag, int_size, 1, fp);
		fwrite(&B, int_size, 1, fp);
		fwrite(&T, int_size, 1, fp);
		fwrite(&C, int_size, 1, fp);
		fwrite(&D, int_size, 1, fp);
		fwrite(&r1, int_size, 1, fp);
		fwrite(&r2, int_size, 1, fp);
		fwrite(&r3, int_size, 1, fp);
		fwrite(&nl, int_size, 1, fp);
		fwrite(&rx, int_size, 1, fp);
		fwrite(&rt, int_size, 1, fp);
		fwrite(&rb, int_size, 1, fp);
		fwrite(&docyl, int_size, 1, fp);
		fwrite(&dk, int_size, 1, fp);

		//Print AMR grid hierarchy
		for (u = 0; u <= n_max; u++){
			fwrite(&block[u][AMR_REFINED], int_size, 1, fp);
		}
		for (u = 0; u <= n_max; u++){
			fwrite(&block[u][AMR_ACTIVE], int_size, 1, fp);
		}
	}
}

void dump_block(MPI_File *fp, int n)
{
	int i, j, z, k;
	struct of_geom geom;
	struct of_state q;
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	int float_size = sizeof(float);
	float p_float[NPR+NDIM], ucon_float[NDIM];
	int NB_print = NB;

	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) {
		array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 0] = (float)p[n][index(n, i, j, z)][0];
		array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 1] = (float)p[n][index(n, i, j, z)][1];

		get_geometry(n, i, j, z, CENT, &geom);
		get_state(p[n][index(n, i, j, z)], &geom, &q);
		
		for (k = 0; k < NDIM; k++) array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + (k + 2)] = (float)q.ucon[k];
		array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 6] = (float)p[n][index(n, i, j, z)][5];
		array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 7] = (float)p[n][index(n, i, j, z)][6];
		array[n][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 8] = (float)p[n][index(n, i, j, z)][7];
	}
	MPI_File_iwrite(fp[0], array[n], 9 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_block[n][0]);
}

void dump_blockdiag(MPI_File *fp, int n)
{
	int i, j, z;
	int int_size = sizeof(int);
	int float_size = sizeof(float);
	float fail1, fail2;
	float divb;
	float diag_float[3];
	int di = (N1>1);
	int dj = (N2>1);
	int dz = (N3>1);

	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) {
		array_diag[n][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 0] = (float)divb_calc(n, i, j, z);
		array_diag[n][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 1] = (float)failimage[n][index(n, i, j, z)][0];
		array_diag[n][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 2] = (float)failimage[n][index(n, i, j, z)][1];
		array_diag[n][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 3] = (float)failimage[n][index(n, i, j, z)][2];
	}
	MPI_File_iwrite(fp[0], array_diag[n], 4 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_blockdiag[n][0]);
}

void gdump_block(FILE *fp, int n)
{
	int i, j, k, l;
	double X[NDIM];
	double r, th, phi;
	double i_double, j_double, z_double;
	struct of_geom geom;
	struct of_state q;
	double dxdxp[NDIM][NDIM];
	double zero = 0.0;
	int z = 0; //The grid is axysymmetric, so put out only for z=0
	int int_size = sizeof(int);
	int double_size = sizeof(double);

	/***************************************************************
	Write header information :
	***************************************************************/
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1)
	{
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		i_double = (double)i*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_1);
		j_double = (double)j*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_2);
		z_double = (double)z*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_3);
		fwrite(&i_double, double_size, 1, fp);
		fwrite(&j_double, double_size, 1, fp);
		fwrite(&z_double, double_size, 1, fp);
		fwrite(&X[1], double_size, 1, fp);
		fwrite(&X[2], double_size, 1, fp);
		fwrite(&X[3], double_size, 1, fp);
		fwrite(&r, double_size, 1, fp);
		fwrite(&th, double_size, 1, fp);
		fwrite(&phi, double_size, 1, fp);
	}

	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n])
	{
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);

		get_geometry(n, i, j, z, CENT, &geom);
		dxdxp_func(X, dxdxp);

		//g_{kl}
		for (k = 0; k < NDIM; k++){
			for (l = 0; l < NDIM; l++){
				fwrite(&(geom.gcov[k][l]), double_size, 1, fp);
			}
		}
		//g^{kl}
		for (k = 0; k < NDIM; k++){
			for (l = 0; l < NDIM; l++){
				fwrite(&(geom.gcon[k][l]), double_size, 1, fp);
			}
		}
		//(-deg(g))**0.5
		fwrite(&(geom.g), double_size, 1, fp);

		//dr^i/dx^j
		for (k = 0; k < NDIM; k++) {
			for (l = 0; l < NDIM; l++) {
				fwrite(&(dxdxp[k][l]), double_size, 1, fp);
			}
		}
	}
}


size_t write_to_dump( int is_dry_run, FILE *fp, double *buf, double val )
{
    if (is_dry_run) {
        return(1L);
    }
    if (fp) {
        return fwrite( &val, sizeof(double), 1, fp );
    }
    if (buf) {
        *buf = val;
        return(1L);
    }
    return(0L);
}
