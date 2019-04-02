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

/* restart functions; restart_init and restart_dump */
#include "decs_MPI.h"

void restart_read_grid(void);
int AMR_coord_linear_old(int level, int i, int j, int z);

/*Write restart file*/
void restart_write(void)
{
	int n;
	char filename[100], dirpath[100];
	int int_size = sizeof(int);
	FILE *checkfile;
	int zero = 0;

	//First close rdump files in progress
	close_rdump();

	if (rank == 0){
		if (rdump_cnt % 2 == 0) {
			sprintf(filename, "rdumps0/parameter");
			checkfile = fopen("rdumps0/checkfile", "wb");
			fwrite(&zero, int_size, 1, checkfile);
			fclose(checkfile);
		}
		else {
			sprintf(filename, "rdumps1/parameter");
			checkfile = fopen("rdumps1/checkfile", "wb");
			fwrite(&zero, int_size, 1, checkfile);
			fclose(checkfile);
		}
		fparam_restart = fopen(filename, "wb");	
		dump_params(fparam_restart);
		fflush_unlocked(fparam_restart);
	}

	if (rank == 1%numtasks){
		if (rdump_cnt % 2 == 0) sprintf(filename, "rdumps0/grid");
		else sprintf(filename, "rdumps1/grid");
		grid_restart = fopen(filename, "wb");
		gdump_grid(grid_restart);
		fflush_unlocked(grid_restart);
	}

	for (n = 0; n < n_active; n++){
		if (rdump_cnt % 2 == 0) sprintf(filename, "rdumps0/rdump%d", n_ord[n]);
		else sprintf(filename, "rdumps1/rdump%d", n_ord[n]);
		MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &rdump[nl[n_ord[n]]]);
		rdump_block_write(&rdump[nl[n_ord[n]]], n_ord[n]);
	}
	first_rdump = 1;

	if (rank == 0)fprintf(stderr, "Restart write to %s complete!\n", filename);
	rdump_cnt++;
}

void rdump_block_write(MPI_File *fp, int n)
{
	int i, j, z, k;
	#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G){
		for (k = 0; k < NPR; k++) array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (k)] = p[nl[n]][index_3D(n, i, j, z)][k];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (0 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][0];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (1 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (2 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (3 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3];	
	}
	MPI_File_iwrite_all(fp[0], array_rdump[nl[n]], (NPR + NDIM) * (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G), MPI_DOUBLE, &req_block_rdump[nl[n]][0]);
}

/*Read restart file*/
int restart_read(void)
{
	int n, num;
	char filename[100], dirpath[100];
	FILE *rdump;
	int int_size = sizeof(int);

	//From new grid to old grid to read rdumps1
	for (n = 0; n < n_active; n++){
		num = n_ord[n];
		#if(READ_OLD)
		num = AMR_coord_linear_old(block[num][AMR_LEVEL], block[num][AMR_COORD1], block[num][AMR_COORD2], block[num][AMR_COORD3]);
		#endif	
		if(restart_number==1) sprintf(filename, "rdumps1/rdump%d", num);
		else if (restart_number == 0)sprintf(filename, "rdumps0/rdump%d", num);
		else return 0;

		rdump = fopen(filename, "rb");
		if (rdump == NULL) {
			if (rank == 0) fprintf(stderr, "Cannot open restart file %s\n", filename);
			return 0;
		}
		rdump_block_read(rdump, n_ord[n]);
		fclose(rdump);
	}
	if (n_active == 0) {
		if (rank == 0) fprintf(stderr, "No active blocks in rdump file %s\n", filename);
		return 0;
	}
	/*Disable injection of matter after restart for elliptical orbits*/
	#if (ELLIPTICAL2)
	sourceflag = 0.;
	#endif

	#if( DO_FONT_FIX ) 
	set_Katm();
	#endif 

	if (rank == 0){
		fprintf(stderr, "done with restart init %s \n", filename);
	}

	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif

	/* bound */
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	#endif
	#if(GPU_ENABLED || GPU_DEBUG )
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
	#endif
	return 1;
}

void rdump_block_read(FILE *fp, int n)
{
	int i, j, z, k;
	int double_size = sizeof(double);

	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		PLOOP fread(&(p[nl[n]][index_3D(n, i, j, z)][k]), double_size, 1, fp);
		#if(STAGGERED)
		for (k = 0; k<NDIM; k++) fread(&(ps[nl[n]][index_3D(n, i, j, z)][k]), double_size, 1, fp);
		ps[nl[n]][index_3D(n, i, j, z)][1] /= gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
		ps[nl[n]][index_3D(n, i, j, z)][2] /= gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
		ps[nl[n]][index_3D(n, i, j, z)][3] /= gdet[nl[n]][index_2D(n, i, j, z)][FACE3];
		#endif
		p[nl[n]][index_3D(n, i, j, z)][B1] *= 1.0;
		p[nl[n]][index_3D(n, i, j, z)][B2] *= 1.0;
		p[nl[n]][index_3D(n, i, j, z)][B3] *= 1.0;
	}
}


void close_rdump(void) {
	int u, n;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total % u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;
	FILE *checkfile;
	int one = 1;
	int int_size = sizeof(int);

	//First close rdump files in progress
	if (first_rdump == 1) {
		for (n = 0; n < n_active; n++) {
			MPI_Wait(&req_block_rdump[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
			MPI_File_close(&rdump[nl[n_ord[n]]]);
		}
	}

	if (rank == 0 && fparam_restart != NULL)fclose(fparam_restart);
	if (rank == 1%numtasks && grid_restart != NULL)fclose(grid_restart);

	//Now tell the writing is complete
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		if ((rdump_cnt - 1) % 2 == 0) checkfile = fopen("rdumps0/checkfile", "wb");
		else checkfile = fopen("rdumps1/checkfile", "wb");
		fwrite(&one, int_size, 1, checkfile);
		fclose(checkfile);
	}
	first_rdump = 0;
}


int restart_read_param(void)
{
	int n;
	char filename[100], dirpath[100];
	FILE *param, *checkfile;
	int int_size = sizeof(int);
	double t0=-10.0, t1=-10.0;
	int value0=0, value1=0;
	restart_number = -1;
	
	checkfile = fopen("rdumps0/checkfile", "rb");
	if (checkfile != NULL) {
		fread(&value0, int_size, 1, checkfile);
		fclose(checkfile);
	}

	if (value0 == 1) {
		sprintf(filename, "rdumps0/parameter");
		param = fopen(filename, "rb");
		if (param != NULL) {
			param_read(param);
			fclose(param);
			t0 = t;
			restart_number = 0;
		}
	}

	checkfile = fopen("rdumps1/checkfile", "rb");
	if (checkfile != NULL) {
		fread(&value1, int_size, 1, checkfile);
		fclose(checkfile);
	}

	if (value1 == 1) {
		sprintf(filename, "rdumps1/parameter");
		param = fopen(filename, "rb");
		if (param != NULL) {
			param_read(param);
			fclose(param);
			t1 = t;
			if (t1 > t0) {
				fprintf(stderr, "Reading in rdumps1! \n");
				restart_number = 1;
			}
		}
	}

	if (t0 > t1 && value0==1) {
		sprintf(filename, "rdumps0/parameter");
		fprintf(stderr, "Reading in rdumps0! \n");
		param = fopen(filename, "rb");
		if (param != NULL) {
			param_read(param);
			fclose(param);
			restart_number = 0;
		}
	}
	
	if (restart_number == -1) {
		fprintf(stderr, "No restart dump available! \n");
		return 0;
	}
	else {
		return 1;
	}
}

void param_read(FILE *fp){
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	int u;
	double dummy;
	
	 //Print out essential stuff for restart
	fread(&t, double_size, 1, fp);
	fread(&n_active, int_size, 1, fp);
	fread(&n_active_total, int_size, 1, fp);
	fread(&nstep, int_size, 1, fp);
	fread(&DTd, double_size, 1, fp);
	fread(&DTl, double_size, 1, fp);
	fread(&dummy, double_size, 1, fp);
	fread(&dump_cnt, int_size, 1, fp);
	fread(&rdump_cnt, int_size, 1, fp);
	fread(&dt, double_size, 1, fp);
	fread(&failed, int_size, 1, fp);

	//Print out stuff that should be checked later
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

	fread(&BS1_print, int_size, 1, fp);
	fread(&BS2_print, int_size, 1, fp);
	fread(&BS3_print, int_size, 1, fp);
	fread(&NB_print, int_size, 1, fp);
	fread(&NB1_print, int_size, 1, fp);
	fread(&NB2_print, int_size, 1, fp);
	fread(&NB3_print, int_size, 1, fp);
	fread(&startx[1], double_size, 1, fp);
	fread(&startx[2], double_size, 1, fp);
	fread(&startx[3], double_size, 1, fp);
	fread(&dx[0][1], double_size, 1, fp);
	fread(&dx[0][2], double_size, 1, fp);
	fread(&dx[0][3], double_size, 1, fp);
	fread(&tf, double_size, 1, fp);
	fread(&a, double_size, 1, fp);
	fread(&gam, double_size, 1, fp);
	fread(&cour, double_size, 1, fp);
	fread(&Rin, double_size, 1, fp);
	fread(&Rout, double_size, 1, fp);
	fread(&R0, double_size, 1, fp);
	fread(&fractheta, double_size, 1, fp);
	fread(&lim, int_size, 1, fp);
	fread(&stag, int_size, 1, fp);
	fread(&B, int_size, 1, fp);
	fread(&T, int_size, 1, fp);
	fread(&C, int_size, 1, fp);
	fread(&D, int_size, 1, fp);
	fread(&r1, int_size, 1, fp);
	fread(&r2, int_size, 1, fp);
	fread(&r3, int_size, 1, fp);
	fread(&nl, int_size, 1, fp);
	fread(&rx, int_size, 1, fp);
	fread(&rt, int_size, 1, fp);
	fread(&rb, int_size, 1, fp);
	fread(&docyl, int_size, 1, fp);
	fread(&dk, int_size, 1, fp);

	if (BS1_print != BS_1 || BS2_print != BS_2 || BS3_print != BS_3 || NB1_print != NB_1
		|| NB2_print != NB_2 || NB3_print != NB_3 || stag != STAGGERED
		|| B != BRAVO || T != TANGO || C != CHARLIE || D != DELTA || r1 != REF_1 || r2 != REF_2
		|| r3 != REF_3 || nl != N_LEVELS || rx != RADEXP || rt != RTRANS || rb != RB || docyl != DOCYLINDRIFYCOORDS
		|| dk != DOKTOT){
		if(rank==0) fprintf(stderr, "Error reading in input paramters. Your code will probably segfault. Make sure the restart file is compatible with the present code and grid parameters! \n");
	}
	//Read AMR grid hierarchy
	#if(!READ_OLD)
	for (u = 0; u <= n_max; u++){
		fread(&block[u][AMR_TIMELEVEL], int_size, 1, fp);
	}
	for (u = 0; u <= n_max; u++){
		fread(&block[u][AMR_ACTIVE], int_size, 1, fp);
	}
	#endif
	nstep = 0;
}

//Used to upscale old grid with 3 levels of AMR, to new grid with 4 levels of AMR
void restart_read_grid(void)
{
	int n, k;
	int int_size = sizeof(int);
	int block_read[36];

	char filename[100], dirpath[100];
	FILE *param;

	sprintf(filename, "rdumps1/grid");
	param = fopen(filename, "rb");

	if (param == NULL) {
		if (rank == 0) fprintf(stderr, "Cannot open restart param file\n");
		return;
	}

	fread(&k, int_size, 1, param);
	for (n = 0; n < (NB_1*NB_2*NB_3*(8 * (8 + 1) + 1)); n++) {
		for (k = 0; k < 36; k++) { //SASMARK: why is 36 hard-coded?
			fread(&(block_read[k]), int_size, 1, param);
		}
		block[AMR_coord_linear(block_read[AMR_LEVEL], block_read[AMR_COORD1], block_read[AMR_COORD2], block_read[AMR_COORD3])][AMR_ACTIVE] = block_read[AMR_ACTIVE];
		block[AMR_coord_linear(block_read[AMR_LEVEL], block_read[AMR_COORD1], block_read[AMR_COORD2], block_read[AMR_COORD3])][AMR_REFINED] = block_read[AMR_REFINED];
	}
	fclose(param);
}

int AMR_coord_linear_old(int level, int i, int j, int z){
	int index[N_LEVELS], coord[NDIM], factor[N_LEVELS], u, y, n;

	if (i < 0 || j < 0 || z < 0){
		n = -1;
		return n;
	}

	for (y = 0; y < N_LEVELS - 1; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 2; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2 + REF_3) + 1;
		}
	}
	#if(REVERSE_ORDERING)
	index[0] = ((z - z % (int)pow(1 + REF_3, level)) / pow(1 + REF_3, level) * NB_2*NB_1 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level) * NB_1
		+ (i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level));
	n = index[0] * factor[0];
	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		coord[3] = (z % (int)(pow(1 + REF_3, level - u + 1)) - (z % (int)pow(1 + REF_3, level - u))) / pow(2, level - u);
		index[u] = coord[3] * (1 + REF_2)*(1 + REF_1) + coord[2] * (1 + REF_1) + coord[1]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u] + 1;
	}
	#else
	index[0] = ((i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level) * NB_2*NB_3 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level) * NB_3
		+ (z - z % (int)pow(1 + REF_3, level)) / pow(1 + REF_3, level));
	n = index[0] * factor[0];
	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		coord[3] = (z % (int)(pow(1 + REF_3, level - u + 1)) - (z % (int)pow(1 + REF_3, level - u))) / pow(2, level - u);
		index[u] = coord[1] * (1 + REF_2)*(1 + REF_3) + coord[2] * (1 + REF_3) + coord[3]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u] + 1;
	}
	#endif

	return n;
}

