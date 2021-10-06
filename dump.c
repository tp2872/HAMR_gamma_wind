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
	FILE *file;

	//First close dump files in progress
	close_dump();
	first_dump = 1;

	if (rank == 0) {
		sprintf(dirpath, "mkdir dumps%d", dump_cnt);
		system(dirpath);
	}
	MPI_Barrier(mpi_cartcomm);

	if (rank == 0) {
		sprintf(filename, "dumps%d/parameters", dump_cnt);
		fparam_dump = fopen(filename, "wb");
		dump_params(fparam_dump, 0);
		fflush(fparam_dump);
	}

	sprintf(filename, "dumps%d/new_dump%d", dump_cnt, rank);
	if ((file = fopen(filename, "r")))
	{
		fclose(file);
		remove(filename);
	}
	MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL, &fdump[0]);
	//for (n = 0; n < n_active_total; n++){
	//	for(u=0; u<u_max; u++)if(n>=u*u_stride && n<(u+1)*u_stride){
	//		if (block[n_ord_total[n]][AMR_NODE]==rank){
	//			MPI_File_seek(fdump[u], (n-u*u_stride) * NPRDUMP * BS_1*BS_2*BS_3*sizeof(float), MPI_SEEK_SET);
	//			dump_block(&fdump[u], n_ord_total[n]);
	//		}
	//	}
	//}
	for (n = 0; n < n_active; n++) {
		MPI_File_seek(fdump[0], (n) * NPRDUMP * BS_1 * BS_2 * BS_3 * sizeof(float), MPI_SEEK_SET);
		dump_block(&fdump[0], n_ord[n]);
	}

	#if(DUMP_DIAG)
	if (dump_cnt% DUMP_DIAG_FREQUENCY ==0){
		sprintf(filename, "dumps%d/new_dumpdiag%d", dump_cnt, rank);
		if ((file = fopen(filename, "r")))
		{
			fclose(file);
			remove(filename);
		}
		MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fdumpdiag[0]);

		for (n = 0; n < n_active; n++){
			MPI_File_seek(fdumpdiag[0], (n) * NDIAG * BS_1*BS_2*BS_3*sizeof(float), MPI_SEEK_SET);
			dump_blockdiag(&fdumpdiag[0], n_ord[n]);
		}
	}
	#endif
	dump_cnt++;
}

void dump_new_reduced(void) {
	int n, u;
	char filename[100], dirpath[100];
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;
	FILE *file;
	//First close dump files in progress
	close_dump_reduced();
	first_dump_reduced = 1;

	if (rank == 0 % numtasks) {
		#if defined(_WIN32)
		sprintf(dirpath, "mkdir reduced\\dumps%d", dump_cnt_reduced);
		#else
		sprintf(dirpath, "mkdir -p reduced/dumps%d", dump_cnt_reduced);
		#endif
		system(dirpath);
	}
	MPI_Barrier(mpi_cartcomm);
	
	if (rank == 0 % numtasks) {
		sprintf(filename, "reduced/dumps%d/parameters", dump_cnt_reduced);
		fparam_dump_reduced = fopen(filename, "wb");
		dump_params(fparam_dump_reduced,1);
		fflush(fparam_dump_reduced);
	}

	sprintf(filename, "reduced/dumps%d/new_dump%d", dump_cnt_reduced, rank);
	if ((file = fopen(filename, "r")))
	{
		fclose(file);
		remove(filename);
	}

	MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fdump_reduced[0]);
	//for (n = 0; n < n_active_total; n++) {
	//	for (u = 0; u<u_max; u++)if (n >= u*u_stride && n<(u + 1)*u_stride) {
	//		if (block[n_ord_total[n]][AMR_NODE] == rank) {
	//			MPI_File_seek(fdump_reduced[u], (n - u*u_stride) * NPRDUMP * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 * sizeof(float), MPI_SEEK_SET);
	//			dump_block_reduced(&fdump_reduced[u], n_ord_total[n]);
	//		}
	//	}
	//}
	for (n = 0; n < n_active; n++) {
		MPI_File_seek(fdump_reduced[0], (n) * (NPRDUMP) * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 * sizeof(float), MPI_SEEK_SET);
		dump_block_reduced(&fdump_reduced[0], n_ord[n]);
	}
	dump_cnt_reduced++;
}

void close_dump(void) {
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total % u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	if (first_dump == 1) {
		if (rank == 0 && fparam_dump != NULL)fclose(fparam_dump);

		for (n = 0; n < n_active; n++) {
			MPI_Wait(&req_block[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
			#if(DUMP_DIAG)
			if ((dump_cnt - 1) % DUMP_DIAG_FREQUENCY == 0) {
				MPI_Wait(&req_blockdiag[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][1]);
			}
			#endif
		}

		MPI_File_close(&fdump[0]);
		for (u = 0; u < u_max; u++) {
			//MPI_File_close(&fdump[u]);
			#if(DUMP_DIAG)
			if ((dump_cnt - 1) % DUMP_DIAG_FREQUENCY == 0) {
				MPI_File_close(&fdumpdiag[u]);
			}
			#endif
		}
	}
	first_dump = 0;
}

void close_dump_reduced(void) {
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total % u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	if (first_dump_reduced == 1) {
		if (rank == 0 % numtasks && fparam_dump_reduced != NULL)fclose(fparam_dump_reduced);

		for (n = 0; n < n_active; n++) {
			MPI_Wait(&req_block_reduced[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
		}

		MPI_File_close(&fdump_reduced[0]);
		//for (u = 0; u < u_max; u++) {
			//MPI_File_close(&fdump_reduced[u]);
		//}
	}
	first_dump_reduced = 0;
}

void dump_params(FILE *fp, int dump_reduced)
{
	int u, n;
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	int BS1_print, BS2_print, BS3_print;
	if (!dump_reduced) {
		BS1_print = BS_1;
		BS2_print = BS_2;
		BS3_print = BS_3;
	}
	else {
		BS1_print = BS_1 / REDUCE_FACTOR1;
		BS2_print = BS_2 / REDUCE_FACTOR2;
		BS3_print = BS_3 / REDUCE_FACTOR3;
	}
	int NB_print = NB;
	int NB1_print = NB_1;
	int NB2_print = NB_2;
	int NB3_print = NB_3;
	int stag = STAGGERED;
	double density = MASS_DENSITY_SCALE;
	int B = BRAVO;
	int f1 = REDUCE_FACTOR1;
	int f2 = REDUCE_FACTOR2;
	int f3 = REDUCE_FACTOR3;
	int r1 = REF_1;
	int r2 = REF_2;
	int r3 = REF_3;
	int nl = N_LEVELS;
	int rd = dump_reduced;
	int rt = RTRANS;
	int rb = RB;
	int docyl = RAD_M1 + RESISTIVE * 10 + TWO_T * 100 + P_NUM * 1000 + DO_YE * 10000;
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
	fwrite(&density, double_size, 1, fp);
	fwrite(&lim, int_size, 1, fp);
	fwrite(&stag, int_size, 1, fp);
	fwrite(&dump_cnt_reduced, int_size, 1, fp);
	fwrite(&f1, int_size, 1, fp);
	fwrite(&f2, int_size, 1, fp);
	fwrite(&f3, int_size, 1, fp);
	fwrite(&r1, int_size, 1, fp);
	fwrite(&r2, int_size, 1, fp);
	fwrite(&r3, int_size, 1, fp);
	fwrite(&nl, int_size, 1, fp);
	fwrite(&rd, int_size, 1, fp);
	fwrite(&rt, int_size, 1, fp);
	fwrite(&rb, int_size, 1, fp);
	fwrite(&docyl, int_size, 1, fp);
	fwrite(&dk, int_size, 1, fp);

	//Print AMR grid hierarchy
	for (u = 0; u < numtasks; u++) {
		for (n = 0; n < n_active_node[u]; n++) {
			fwrite(&n_ord_node[u][n], int_size, 1, fp);
			fwrite(&block[n_ord_node[u][n]][AMR_TIMELEVEL], int_size, 1, fp);
			fwrite(&block[n_ord_node[u][n]][AMR_NODE], int_size, 1, fp);
		}
	}
}

void dump_block(MPI_File *fp, int n)
{
	int i, j, z, k;
	double ucon[NDIM], ucon_rad[NDIM];
	struct of_geom geom;

    #pragma omp parallel for collapse(3) schedule(static,(BS_1)*(BS_2)*(BS_3)/nthreads) private(i,j,z,k,geom, ucon, ucon_rad)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
        array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2* BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n])* NPRDUMP + RHO] = (float)p[nl[n]][index_3D(n, i, j, z)][0];
        array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2* BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n])* NPRDUMP + UU] = (float)p[nl[n]][index_3D(n, i, j, z)][1];

		get_geometry(n, i, j, z, CENT, &geom);
		ucon_calc(p[nl[n]][index_3D(n, i, j, z)], &geom, ucon);

		for (k = 0; k < NDIM; k++) array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (U1 + k)] = (float)ucon[k];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (B1 + 1)] = (float)p[nl[n]][index_3D(n, i, j, z)][B1];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (B2 + 1)] = (float)p[nl[n]][index_3D(n, i, j, z)][B2];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (B3 + 1)] = (float)p[nl[n]][index_3D(n, i, j, z)][B3];

		#if(RAD_M1)
		ucon_calc_rad(p[nl[n]][index_3D(n, i, j, z)], &geom, ucon_rad);
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (UU_RAD + !DOKTOT)] = (float)p[nl[n]][index_3D(n, i, j, z)][UU_RAD];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (UU_RAD + !DOKTOT + 1)] = (float)ucon_rad[0];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (UU_RAD + !DOKTOT + 2)] = (float)ucon_rad[1];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (UU_RAD + !DOKTOT + 3)] = (float)ucon_rad[2];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (UU_RAD + !DOKTOT + 4)] = (float)ucon_rad[3];
		#endif

		#if(RESISTIVE)
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (E1 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i, j, z)][E1];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (E2 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i, j, z)][E2];
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (E3 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i, j, z)][E3];
		#endif

		#if(TWO_T)
		double Te, Ti;
		Te = calc_Te(p[nl[n]][index_3D(n, i, j, z)]);
		Ti = calc_Ti(p[nl[n]][index_3D(n, i, j, z)]);
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (ENTRE + !DOKTOT + RAD_M1)] = (float)Te;
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (ENTRI + !DOKTOT + RAD_M1)] = (float)Ti;
		#endif

		#if(P_NUM)
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (PHOTON + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i, j, z)][PHOTON];
		#endif

		#if(DO_YE)
		array[nl[n]][(i - N1_GPU_offset[n]) * NPRDUMP * BS_2 * BS_3 + (j - N2_GPU_offset[n]) * NPRDUMP * BS_3 + (z - N3_GPU_offset[n]) * NPRDUMP + (YE + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i, j, z)][YE];
		#endif
	}
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array[nl[n]], NPRDUMP * BS_1*BS_2*BS_3, MPI_FLOAT, &req_block[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array[nl[n]], NPRDUMP * BS_1*BS_2*BS_3, MPI_FLOAT, &req_block[nl[n]][0]);
	#endif
}

void dump_block_reduced(MPI_File *fp, int n){
	int i, j, z, k;
	int i1, j1, z1;
	struct of_geom geom;
	double ucon[NDIM], ucon_rad[NDIM];
	float factor = 1.0;// / ((double)(REDUCE_FACTOR1*REDUCE_FACTOR2*REDUCE_FACTOR3));

    #pragma omp parallel for collapse(3) schedule(static,(BS_1 / REDUCE_FACTOR1)*(BS_2 / REDUCE_FACTOR2)*(BS_3 / REDUCE_FACTOR3)/nthreads) private(i,j,z,i1,j1,z1,k,geom, ucon,ucon_rad)
	for (i = 0; i < BS_1 / REDUCE_FACTOR1; i++)for (j = 0; j < BS_2 / REDUCE_FACTOR2; j++)for (z = 0; z < BS_3 / REDUCE_FACTOR3; z++) {
        for (k = 0; k < 9; k++) array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + k] = 0;
		for (i1 = 0; i1 < 1; i1++)for (j1 = 0; j1 < 1; j1++)for (z1 = 0; z1 < 1; z1++) {
			array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z) * NPRDUMP + RHO] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][0] * factor;
			array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z) * NPRDUMP + UU] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][1] * factor;

			get_geometry(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n], CENT, &geom);
			ucon_calc(p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])], &geom, ucon);

			for (k = 0; k < NDIM; k++) array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (U1 + k)] = (float)ucon[k] * factor;
			array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z) * NPRDUMP + 6] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][5] * factor;
			array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z) * NPRDUMP + 7] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][6] * factor;
			array_reduced[nl[n]][(i) * NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z) * NPRDUMP + 8] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][7] * factor;
            
            #if(RAD_M1)
			ucon_calc_rad(p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])], &geom, ucon_rad);
			array_reduced[nl[n]][(i)* NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j)* NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + (UU_RAD + !DOKTOT)] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j*REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z*REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][UU_RAD] * factor;
            array_reduced[nl[n]][(i)* NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j)* NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + (UU_RAD + !DOKTOT + 1)] = (float)ucon_rad[0] * factor;
            array_reduced[nl[n]][(i)* NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j)* NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + (UU_RAD + !DOKTOT + 2)] = (float)ucon_rad[1] * factor;
            array_reduced[nl[n]][(i)* NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j)* NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + (UU_RAD + !DOKTOT + 3)] = (float)ucon_rad[2] * factor;
            array_reduced[nl[n]][(i)* NPRDUMP * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j)* NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)* NPRDUMP + (UU_RAD + !DOKTOT + 4)] = (float)ucon_rad[3] * factor;
            #endif

			#if(TWO_T)
			double Te, Ti;
			Te = calc_Te(p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])]);
			Ti = calc_Ti(p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])]);
			array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (ENTRE + !DOKTOT + RAD_M1)] = (float)Te * factor;
			array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (ENTRI + !DOKTOT + RAD_M1)] = (float)Ti * factor;
			#endif

			#if(RESISTIVE)
			array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (E1 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][E1] * factor;
			array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (E2 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][E2] * factor;
			array_reduced[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (E3 + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][E3] * factor;
			#endif

			#if(P_NUM)
			array[nl[n]][(i)*NPRDUMP * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j)*NPRDUMP * BS_3 / REDUCE_FACTOR3 + (z)*NPRDUMP + (PHOTON + !DOKTOT + RAD_M1)] = (float)p[nl[n]][index_3D(n, i * REDUCE_FACTOR1 + i1 + N1_GPU_offset[n], j * REDUCE_FACTOR2 + j1 + N2_GPU_offset[n], z * REDUCE_FACTOR3 + z1 + N3_GPU_offset[n])][PHOTON] * factor;
			#endif
		}
	}
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_reduced[nl[n]], NPRDUMP * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3, MPI_FLOAT, &req_block_reduced[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_reduced[nl[n]], NPRDUMP * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3, MPI_FLOAT, &req_block_reduced[nl[n]][0]);
	#endif
}

void dump_blockdiag(MPI_File *fp, int n)
{
	int i, j, z;
	#pragma omp parallel for collapse(3) schedule(static,(BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads) private(i,j,z)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * NDIAG * BS_2* BS_3 + (j - N2_GPU_offset[n]) * NDIAG * BS_3 + (z - N3_GPU_offset[n]) * NDIAG + 0] = (float)divb_calc(n, i, j, z);
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * NDIAG * BS_2* BS_3 + (j - N2_GPU_offset[n]) * NDIAG * BS_3 + (z - N3_GPU_offset[n]) * NDIAG + 1] = (float)failimage[nl[n]][index_3D(n, i, j, z)][0];
		array_diag[nl[n]][(i - N1_GPU_offset[n]) * NDIAG * BS_2* BS_3 + (j - N2_GPU_offset[n]) * NDIAG * BS_3 + (z - N3_GPU_offset[n]) * NDIAG + 2] = (float)failimage[nl[n]][index_3D(n, i, j, z)][1];
	}
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_diag[nl[n]], NDIAG * BS_1*BS_2*BS_3, MPI_FLOAT, &req_blockdiag[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_diag[nl[n]], NDIAG * BS_1*BS_2*BS_3, MPI_FLOAT, &req_blockdiag[nl[n]][0]);
	#endif
}

void gdump_new(void){
	int n;
	char filename[100];
	
	FILE *grid, *file;
	if (rank == 1 % numtasks && nstep==0){
		sprintf(filename, "gdumps/grid");
		grid = fopen(filename, "wb");
		gdump_grid(grid);
		fclose(grid);
	}

	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][GDUMP_WRITTEN] != 1 && block[n_ord_total[n]][GDUMP_WRITTEN] != 2){
			sprintf(filename, "gdumps/gdump%d", n_ord_total[n]);
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				if ((file = fopen(filename, "r")))
				{
					fclose(file);
					remove(filename);
				}
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

void gdump_new_reduced(void) {
	int n;
	char filename[100];

	FILE *grid;
	if (rank == 1 % numtasks && nstep == 0) {
		sprintf(filename, "reduced/gdumps/grid");
		grid = fopen(filename, "wb");
		gdump_grid(grid);
		fclose(grid);
	}

	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] != 1 && block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] != 2) {
			sprintf(filename, "reduced/gdumps/gdump%d", n_ord_total[n]);
			if (block[n_ord_total[n]][AMR_NODE] == rank) {
				MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &gdump_reduced[nl[n_ord_total[n]]]);
				gdump_block_reduced(&gdump_reduced[nl[n_ord_total[n]]], n_ord_total[n]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] = 2;
		}
		else if (block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] == 2) {
			if (block[n_ord_total[n]][AMR_NODE] == rank) {
				MPI_Wait(&req_gdump1_reduced[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_Wait(&req_gdump2_reduced[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_File_close(&gdump_reduced[nl[n_ord_total[n]]]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] = 1;
		}
	}
}

void gdump_grid(FILE *fp)
{
	int n, k;
	array_gdumpgrid[0] = NB;

	fwrite(&array_gdumpgrid[0], sizeof(int), 1, fp);
	for (n = 0; n <= n_max; n++){
		for (k = 0; k < NV; k++){
			fwrite(&block[n][k],sizeof(int), 1, fp);
		}
	}
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

	#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z,k,X,r,th,phi)
	ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1)
	{
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);

		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 0] = (double)i*pow(2., (N_LEVELS - (block[n][AMR_LEVEL1] + 1))*REF_1);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 1] = (double)j*pow(2., (N_LEVELS - (block[n][AMR_LEVEL2] + 1))*REF_2);
		array_gdump1[nl[n]][(i - N1_GPU_offset[n]) * 9 * BS_2* BS_3 + (j - N2_GPU_offset[n]) * 9 * BS_3 + (z - N3_GPU_offset[n]) * 9 + 2] = (double)z*pow(2., (N_LEVELS - (block[n][AMR_LEVEL3] + 1))*REF_3);
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

	#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2)/nthreads) private(i,j,z,k,l,X,geom,dxdxp)
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

void gdump_block_reduced(MPI_File  *fp, int n)
{
	int i, j, k, l, i1, j1, z1;
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
	float factor = 1.0;// / ((double)(REDUCE_FACTOR1*REDUCE_FACTOR2*REDUCE_FACTOR3));

	#pragma omp parallel for collapse(3) schedule(static,(BS_1 / REDUCE_FACTOR1)*(BS_2 / REDUCE_FACTOR2)*(BS_3 / REDUCE_FACTOR3)/nthreads) private(i,j,z,i1,j1,z1,k,geom,q, r, th, phi, X)
	for (i = 0; i < BS_1 / REDUCE_FACTOR1; i++)for (j = 0; j < BS_2 / REDUCE_FACTOR2; j++)for (z = 0; z < BS_3 / REDUCE_FACTOR3; z++) {
		for (k = 0; k < 9; k++)array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2* BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + k] = 0.;
		for (i1 = 0; i1 < 1; i1++)for (j1 = 0; j1 < 1; j1++)for (z1 = 0; z1 < 1; z1++) {
			coord(n, N1_GPU_offset[n] + i*REDUCE_FACTOR1 + i1, N2_GPU_offset[n] + j*REDUCE_FACTOR2 + j1, N3_GPU_offset[n] + z*REDUCE_FACTOR3 + z1, CENT, X);
			bl_coord(X, &r, &th, &phi);
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 0] = (double)i*REDUCE_FACTOR1*pow(2., (N_LEVELS - (block[n][AMR_LEVEL1] + 1))*REF_1) * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 1] = (double)j*REDUCE_FACTOR2*pow(2., (N_LEVELS - (block[n][AMR_LEVEL2] + 1))*REF_2) * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 2] = (double)z*REDUCE_FACTOR3*pow(2., (N_LEVELS - (block[n][AMR_LEVEL3] + 1))*REF_3) * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 3] = X[1] * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 4] = X[2] * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 5] = X[3] * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 6] = r * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 7] = th * factor;
			array_gdump1_reduced[nl[n]][(i) * 9 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 + (j) * 9 * BS_3 / REDUCE_FACTOR3 + (z) * 9 + 8] = phi * factor;
		}
	}
	MPI_File_seek(fp[0], 0 * sizeof(double), MPI_SEEK_SET);
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_gdump1_reduced[nl[n]], 9 * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3, MPI_DOUBLE, &req_gdump1_reduced[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_gdump1_reduced[nl[n]], 9 * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3, MPI_DOUBLE, &req_gdump1_reduced[nl[n]][0]);
	#endif

	if (rank == 0 && NSY)fprintf(stderr, "Warning: metric is written out unsymmetric! \n");

	factor = 1.0;// / ((double)(REDUCE_FACTOR1*REDUCE_FACTOR2));
	#pragma omp parallel for collapse(2) schedule(static,(BS_1*BS_2)/nthreads) private(i,j,z,i1,j1,z1,k,l,X,geom,dxdxp)
	for (i = 0; i < BS_1 / REDUCE_FACTOR1; i++)for (j = 0; j < BS_2 / REDUCE_FACTOR2; j++) {
		z = 0;
		for (k = 0; k < 49; k++)array_gdump2_reduced[nl[n]][(i) * 49 * BS_2 / REDUCE_FACTOR2 + (j) * 49 + k] = 0.;
		for (i1 = 0; i1 < 1; i1++)for (j1 = 0; j1 < 1; j1++) {

			coord(n, N1_GPU_offset[n] + i*REDUCE_FACTOR1 + i1, N2_GPU_offset[n] + j*REDUCE_FACTOR2 + j1, N3_GPU_offset[n] + z, CENT, X);
			get_geometry(n, N1_GPU_offset[n] + i*REDUCE_FACTOR1 + i1, N2_GPU_offset[n] + j*REDUCE_FACTOR2 + j1, N3_GPU_offset[n] + z, CENT, &geom);
			dxdxp_func(X, dxdxp);

			//g_{kl}
			for (k = 0; k < NDIM; k++) {
				for (l = 0; l < NDIM; l++) {
					array_gdump2_reduced[nl[n]][(i) * 49 * BS_2 / REDUCE_FACTOR2 + (j) * 49 + k*NDIM + l] = geom.gcov[k][l] * factor;
				}
			}
			//g^{kl}
			for (k = 0; k < NDIM; k++) {
				for (l = 0; l < NDIM; l++) {
					array_gdump2_reduced[nl[n]][(i) * 49 * BS_2 / REDUCE_FACTOR2 + (j) * 49 + NDIM*NDIM + k*NDIM + l] = geom.gcon[k][l] * factor;
				}
			}
			//(-deg(g))**0.5
			array_gdump2_reduced[nl[n]][(i) * 49 * BS_2 / REDUCE_FACTOR2 + (j) * 49 + 2 * NDIM*NDIM] = geom.g * factor;

			//dr^i/dx^j
			for (k = 0; k < NDIM; k++) {
				for (l = 0; l < NDIM; l++) {
					array_gdump2_reduced[nl[n]][(i) * 49 * BS_2 / REDUCE_FACTOR2 + (j) * 49 + 2 * NDIM*NDIM + 1 + k*NDIM + l] = dxdxp[k][l] * factor;
				}
			}
		}
	}
	MPI_File_seek(fp[0], 9 * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 * sizeof(double), MPI_SEEK_SET);
	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_gdump2_reduced[nl[n]], 49 * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2, MPI_DOUBLE, &req_gdump2_reduced[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_gdump2_reduced[nl[n]], 49 * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2, MPI_DOUBLE, &req_gdump2_reduced[nl[n]][0]);
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

void close_gdump_reduced(void) {
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] == 2) {
			if (block[n_ord_total[n]][AMR_NODE] == rank) {
				MPI_Wait(&req_gdump1_reduced[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_Wait(&req_gdump2_reduced[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_File_close(&gdump_reduced[nl[n_ord_total[n]]]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN_REDUCED] = 1;
		}
	}
}
