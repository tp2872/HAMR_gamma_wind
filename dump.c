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

void dump_new(void){
	int n, u;
	char filename[100], dirpath[100];
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	//First close dump files in progress
	close_dump();

	if (rank == 0){
		sprintf(dirpath, "mkdir dumps%d", dump_cnt);
		system(dirpath);
		sprintf(filename, "dumps%d/parameters", dump_cnt);
		fparam_dump = fopen(filename, "wb");
		dump_params(fparam_dump);
		fflush(fparam_dump);
	}

	first_dump = 1;

	if (rank == (1 % numtasks)) {
		sprintf(filename, "dumps%d/grid", dump_cnt);
		MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &grid_dump[0]);
		gdump_grid(grid_dump);
	}

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

	if (dump_cnt%10==0){
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
	dump_cnt++;
}

void close_dump(void) {
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total % u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	if (first_dump == 1) {
		if (rank == 0 && fparam_dump != NULL)fclose(fparam_dump);
		//if (rank == 1 % numtasks && grid_dump != NULL)fclose(grid_dump);

		for (n = 0; n < n_active; n++) {
			MPI_Wait(&req_block[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
			if ((dump_cnt - 1) % 10 == 0) {
				MPI_Wait(&req_blockdiag[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][1]);
			}
		}

		for (u = 0; u < u_max; u++) {
			MPI_File_close(&fdump[u]);
			if ((dump_cnt - 1) % 10 == 0) {
				MPI_File_close(&fdumpdiag[u]);
			}
		}
		if (rank == 1 % numtasks) {
			MPI_Wait(&req_gdumpgrid[0], &Statbound[nl[n_ord[0]]][0]);
			MPI_File_close(&grid_dump[0]);
		}
	}
	first_dump = 0;
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

		////Print AMR grid hierarchy
		//for (u = 0; u <= n_max; u++){
		//	fwrite(&block[u][AMR_TIMELEVEL], int_size, 1, fp);
		//}
		//for (u = 0; u <= n_max; u++){
		//	fwrite(&block[u][AMR_ACTIVE], int_size, 1, fp);
		//}
	}
}

void dump_block(MPI_File *fp, int n)
{
	int i, j, z, k;
	struct of_geom geom;
	struct of_state q;

	#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,geom,q)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
		array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 0] = (float)p[nl[n]][index_3D(n, i, j, z)][0];
		array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 1] = (float)p[nl[n]][index_3D(n, i, j, z)][1];

		get_geometry(n, i, j, z, CENT, &geom);
		get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);

		for (k = 0; k < NDIM; k++) array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + (k + 2)] = (float)q.ucon[k];
		array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 6] = (float)p[nl[n]][index_3D(n, i, j, z)][5];
		array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 7] = (float)p[nl[n]][index_3D(n, i, j, z)][6];
		array[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 8] = (float)p[nl[n]][index_3D(n, i, j, z)][7];
	}
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array[nl[n]], 9 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_block[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array[nl[n]], 9 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_block[nl[n]][0]);
	#endif
}

void dump_blockdiag(MPI_File *fp, int n)
{
	int i, j, z;
	#pragma omp parallel for schedule(dynamic,1) private(i,j,z)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 0] = (float)divb_calc(n, i, j, z);
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 1] = (float)failimage[nl[n]][index_3D(n, i, j, z)][0];
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 2] = (float)failimage[nl[n]][index_3D(n, i, j, z)][1];
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * 4 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 4 * BS_3 + (z - N3_GPU_offset[n]) * 4 + 3] = (float)failimage[nl[n]][index_3D(n, i, j, z)][2];
	}
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_diag[nl[n]], 4 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_blockdiag[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_diag[nl[n]], 4 * BS_1*BS_2*BS_3, MPI_FLOAT, &req_blockdiag[nl[n]][0]);
	#endif
}

void gdump_new(void){
	int n;
	char filename[100];

	//FILE *grid;
	//if (rank == 0 && nstep==0){
	//	sprintf(filename, "gdumps/grid");
	//	grid = fopen(filename, "wb");
	//	gdump_grid(grid);
	//	fclose(grid);
	//}

	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][GDUMP_WRITTEN] != 1 && block[n_ord_total[n]][GDUMP_WRITTEN] != 2){
			sprintf(filename, "gdumps/gdump%d", n_ord_total[n]);
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &gdump[nl[n_ord_total[n]]]);
				gdump_block(&gdump[nl[n_ord_total[n]]], n_ord_total[n]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN] = 2;
		}
		else if(block[n_ord_total[n]][GDUMP_WRITTEN] == 2){
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				MPI_Wait(&req_gdump1[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_Wait(&req_gdump2[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_File_close(&gdump[nl[n_ord_total[n]]]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN] = 1;
		}
	}
}

void gdump_grid(MPI_File *fp)
{
	int n, k;
	array_gdumpgrid[0] = NB;

	#pragma omp parallel for schedule(dynamic,1) private(n,k)
	for (n = 0; n <= n_max; n++){
		for (k = 0; k < NV; k++){
			array_gdumpgrid[1 + n*NV + k] = block[n][k];
		}
	}

	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_gdumpgrid, 1 + NB*NV, MPI_INT, &req_gdumpgrid[0]);
	#else
	MPI_File_iwrite(fp[0], array_gdumpgrid,1+NB*NV, MPI_INT, &req_gdumpgrid[0]);
	#endif
}

void gdump_block(MPI_File  *fp, int n)
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
	#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,X,r,th,phi)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1)
	{
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 0] = (double)i*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_1);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 1] = (double)j*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_2);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 2] = (double)z*pow(2., (N_LEVELS - (block[n][AMR_LEVEL] + 1))*REF_3);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 3] = X[1];
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 4] = X[2];
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 5] = X[3];
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 6] = r;
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 7] = th;
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 8] = phi;
	}
	MPI_File_seek(fp[0], 0*sizeof(double), MPI_SEEK_SET);
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_gdump1[nl[n]], 9 * BS_1*BS_2*BS_3, MPI_DOUBLE, &req_gdump1[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_gdump1[nl[n]], 9 * BS_1*BS_2*BS_3, MPI_DOUBLE, &req_gdump1[nl[n]][0]);
	#endif

	#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,l,X,geom,dxdxp)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n])
	{
		coord(n, i, j, z, CENT, X);
		get_geometry(n, i, j, z, CENT, &geom);
		dxdxp_func(X, dxdxp);

		//g_{kl}
		for (k = 0; k < NDIM; k++){
			for (l = 0; l < NDIM; l++){
				array_gdump2[nl[n]][(i - N1_GPU_offset[n]) * 49 * BS_2 + (j - N2_GPU_offset[n]) * 49 + k*NDIM + l] = geom.gcov[k][l];
			}
		}
		//g^{kl}
		for (k = 0; k < NDIM; k++){
			for (l = 0; l < NDIM; l++){
				array_gdump2[nl[n]][(i - N1_GPU_offset[n]) * 49 * BS_2 + (j - N2_GPU_offset[n]) * 49 + NDIM*NDIM + k*NDIM + l] = geom.gcon[k][l];
			}
		}
		//(-deg(g))**0.5
		array_gdump2[nl[n]][(i - N1_GPU_offset[n]) * 49 * BS_2 + (j - N2_GPU_offset[n]) * 49 + 2 * NDIM*NDIM] = geom.g;

		//dr^i/dx^j
		for (k = 0; k < NDIM; k++) {
			for (l = 0; l < NDIM; l++) {
				array_gdump2[nl[n]][(i - N1_GPU_offset[n]) * 49 * BS_2 + (j - N2_GPU_offset[n]) * 49 + 2*NDIM*NDIM + 1 + k*NDIM + l] = dxdxp[k][l];
			}
		}
	}
	MPI_File_seek(fp[0], 9 * BS_1*BS_2*BS_3*sizeof(double), MPI_SEEK_SET);
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_gdump2[nl[n]], 49 * BS_1*BS_2, MPI_DOUBLE, &req_gdump2[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_gdump2[nl[n]], 49 * BS_1*BS_2, MPI_DOUBLE, &req_gdump2[nl[n]][0]);
	#endif
}

void close_gdump(void){
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][GDUMP_WRITTEN] == 2){
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				MPI_Wait(&req_gdump1[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_Wait(&req_gdump2[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_File_close(&gdump[nl[n_ord_total[n]]]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN] = 1;
		}
	}
}
