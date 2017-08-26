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

void rdump_block_write(MPI_File *fp, int n)
{
	int i, j, z, k;

	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
		for (k = 0; k < NPR; k++) array_rdump[n][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (k)] = p[n][index(n, i, j, z)][k];
		for (k = 0; k < NDIM; k++) array_rdump[n][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (k + NPR)] = ps[n][index(n, i, j, z)][k];
	}
	MPI_File_iwrite(fp[0], array_rdump[n], (NPR + NDIM) * (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G), MPI_DOUBLE, &req_block_rdump[n][0]);
}

void rdump_block_read(FILE *fp, int n)
{
	int i, j, z, k;
	int double_size = sizeof(double);

	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
		PLOOP fread(&(p[n][index(n, i, j, z)][k]), double_size, 1, fp);
		#if(STAGGERED)
		for (k = 0; k<NDIM; k++) fread(&(ps[n][index(n, i, j, z)][k]), double_size, 1, fp);
		for (k = 0; k<NDIM; k++) ps[n][index(n, i, j, z)][k]*=1.0;
		#endif
 		p[n][index(n, i, j, z)][B1]*=1.0;
		p[n][index(n, i, j, z)][B2]*=1.0;
		p[n][index(n, i, j, z)][B3]*=1.0;
	}
}

void restart_write(void)
{
	int n;
	char filename[100], dirpath[100];
	FILE *param;
	if (rank == 0){
		//sprintf(dirpath, "mkdir rdumps%d", dump_cnt);
		//system(dirpath);
		if (rdump_cnt % 10 == 0) sprintf(filename, "rdumps0/parameter");
		else sprintf(filename, "rdumps1/parameter");
		param = fopen(filename, "wb");
		if (rank == 0) dump_params(param);
		fclose(param);
	}
	for (n = 0; n < n_active; n++){
		if (rdump_cnt % 10 == 0) sprintf(filename, "rdumps0/rdump%d", n_ord[n]);
		else sprintf(filename, "rdumps1/rdump%d",  n_ord[n]);
		MPI_File_open(mpi_self, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &rdump[n_ord[n]]);
		rdump_block_write(&rdump[n_ord[n]], n_ord[n]);
	}
	first_rdump = 1;

	if (rank==0)fprintf(stderr, "Restart write to %s complete!\n", filename);
	rdump_cnt++;
}

void param_read(FILE *fp){
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	int u;

	//Print out essential stuff for restart
	fread(&t, double_size, 1, fp);
	fread(&n_active, int_size, 1, fp);
	fread(&n_active_total, int_size, 1, fp);
	fread(&nstep, int_size, 1, fp);
	fread(&DTd, double_size, 1, fp);
	fread(&DTl, double_size, 1, fp);
	fread(&DTr, double_size, 1, fp);
	fread(&dump_cnt, int_size, 1, fp);
	fread(&rdump_cnt, int_size, 1, fp);
	fread(&dt, double_size, 1, fp);
	fread(&failed, int_size, 1, fp);
	
	if (calc_mem(n_active_total)>((double)numtasks*(double)(N_GPU)* 4. * (pow(10., 9.))) && rank == 0){
		fprintf(stderr, "You are exceeding the maximum memory size of 4 GB per GPU by reading in too many blocks! Code will segfault! \n");
		max_levels -= 1;
	}

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
		fprintf(stderr, "Error reading in input paramters. Your code will probably segfault. Make sure the restart file is compatible with the present code and grid parameters! \n");
	}
	//Read AMR grid hierarchy
	for (u = 0; u <= n_max; u++){
		fread(&block[u][AMR_REFINED], int_size, 1, fp);
	}
	for (u = 0; u <= n_max; u++){
		fread(&block[u][AMR_ACTIVE], int_size, 1, fp);
	}
}

int restart_read(void)
{
	int n;
	char filename[100], dirpath[100];
	FILE *rdump;

	for (n = 0; n < n_active; n++){
		if (rdump_cnt % 2 == 4) sprintf(filename, "rdumps0/rdump%d", n_ord[n]);
		else sprintf(filename, "rdumps1/rdump%d", n_ord[n]);
		rdump = fopen(filename, "rb");
		if (rdump == NULL) {
			if (rank == 0) fprintf(stderr, "Cannot open restart file %s\n", filename);
			return 0;
		}
		rdump_block_read(rdump, n_ord[n]);
		fclose(rdump);
	}
	if (n_active == 0) {
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

	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
	#endif
	return 1;
}

int restart_read_param(void)
{
	int n;
	char filename[100], dirpath[100];
	FILE *param;

	if (rdump_cnt % 2 == 4) sprintf(filename, "rdumps0/parameter");
	else sprintf(filename, "rdumps1/parameter");
	param = fopen(filename, "rb");
	if (param == NULL) {
		if (rank == 0) fprintf(stderr, "Cannot open restart param file\n");
		return 0;
	}
	param_read(param);
	fclose(param);
	return 1;
}


