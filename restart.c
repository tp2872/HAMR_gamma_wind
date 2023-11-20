/* restart functions; restart_init and restart_dump */
#include "include.h"
#include "decs_MPI.h"

//Insert a toroidal field of beta=2 in a thin disk after restart. Works only with axisymmetric AMR for non-tilted disks!
#define INSERT_TOROIDAL (0)

void add_toroidal_B(void);

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
		dump_params(fparam_restart, 0);
		fflush(fparam_restart);
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
	#pragma omp parallel for collapse(3) schedule(static,(BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads) private(i,j,z,k)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G){
		for (k = 0; k < NPR; k++) array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (k)] = p[nl[n]][index_3D(n, i, j, z)][k];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (0 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][0];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (1 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (2 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
		array_rdump[nl[n]][(i - N1_GPU_offset[n] + N1G) * (NPR + NDIM) * (BS_2 + 2 * N2G)* (BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G) * (NPR + NDIM) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G) * (NPR + NDIM) + (3 + NPR)] = ps[nl[n]][index_3D(n, i, j, z)][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3];	
	}

	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_rdump[nl[n]], (NPR + NDIM) * (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G), MPI_DOUBLE, &req_block_rdump[nl[n]][0]);
	#else
	MPI_File_iwrite(fp[0], array_rdump[nl[n]], (NPR + NDIM) * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), MPI_DOUBLE, &req_block_rdump[nl[n]][0]);
	#endif
}

void rdump_grid(MPI_File *fp)
{
	int n, k;
	array_rdumpgrid[0] = NB;

	#pragma omp parallel for schedule(static, NB/nthreads) private(n,k)
	for (n = 0; n <= n_max; n++) {
		for (k = 0; k < NV; k++) {
			array_rdumpgrid[1 + n*NV + k] = block[n][k];
		}
	}

	#if(PARALLEL_IO)
	MPI_File_iwrite_all(fp[0], array_rdumpgrid, 1 + NB*NV, MPI_INT, &req_rdumpgrid[0]);
	#else
	MPI_File_iwrite(fp[0], array_rdumpgrid, 1 + NB*NV, MPI_INT, &req_rdumpgrid[0]);
	#endif
}

/*Read restart file*/
int restart_read(void)
{
	int n, num;
	char filename[100], dirpath[100];
	FILE *rdump, *dummy;
	int int_size = sizeof(int);

	//From new grid to old grid to read rdumps1
	for (n = 0; n < n_active; n++){
		if (restart_number == 1) {
			sprintf(filename, "rdumps1/grid");
			dummy = fopen(filename, "rb");
			if(dummy !=NULL){
				num = n_old[n_ord[n]];
				fclose(dummy);
			}
			else {
				num = n_ord[n];
			}
			sprintf(filename, "rdumps1/rdump%d", num);
		}
		else if (restart_number == 0) {
			sprintf(filename, "rdumps0/grid");
			dummy = fopen(filename, "rb");
			if (dummy != NULL) {
				num = n_old[n_ord[n]];
				fclose(dummy);
			}
			else {
				num = n_ord[n];
			}
			sprintf(filename, "rdumps0/rdump%d", num);
		}
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

	if (rank == 0){
		fprintf(stderr, "done with restart init %s \n", filename);
	}

	#if(INSERT_TOROIDAL)
	bound_prim(p, 1);
	add_toroidal_B();
		#if (MPI_enable)
		MPI_Barrier(mpi_cartcomm);
		#endif
		if (rank == 0) {
			fprintf(stderr, "done with adding magnetic fields %s \n", filename);
		}
	#endif

	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif

	#if(GPU_ENABLED || GPU_DEBUG )
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
	#endif

	/* bound */
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	#endif
	return 1;
}

void rdump_block_read(FILE *fp, int n)
{
	int i, j, z, k, read_geom=0;
	int double_size = sizeof(double);

	int npr_local = NPR_U + read_M1 * NPR_R * RAD_M1 + read_Res * NPR_E * RESISTIVE + read_2T * NPR_2T * TWO_T + read_Pnum * NPR_PH * P_NUM + read_nuclear * DONUCLEAR * 2 + read_Ye * DO_YE * 1 + read_neutrinos * NPR_NU * NEUTRINOS_M1 * NU_SPECIES;
	int npr_file = NPR_U + read_M1 * NPR_R + read_Res * NPR_E + read_2T * NPR_2T + read_Pnum * NPR_PH + read_nuclear * 2 + read_Ye * 1 + read_neutrinos * NPR_NU * NU_SPECIES + NDIM * STAGGERED;
	int red_1, red_2, red_3, i1, j1, z1;
	double reduce_factor;
	double read[NPR_U +  NPR_R * 1 +  NPR_E * 1 + NPR_2T * 1 + NPR_PH * 1 + 1 + 2 + NPR_NU * NU_SPECIES + NDIM * STAGGERED];
	struct of_geom geom;
	#if(RAD_M1)
	int uu_rad = (8 + DOKTOT);
	int u1_rad = (8 + DOKTOT + 1);
	int u2_rad = (8 + DOKTOT + 2);
	int u3_rad = (8 + DOKTOT + 3);
	#endif
	#if(RESISTIVE)
	int e1 = (8 + DOKTOT + read_M1 * 4);
	int e2 = (8 + DOKTOT + read_M1 * 4 + 1);
	int e3 = (8 + DOKTOT + read_M1 * 4 + 2);
	#endif
	#if(TWO_T)
	int entre = (8 + DOKTOT + read_M1 * 4 + read_Res * 3);
	int entri = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + 1);
	#endif
	#if(P_NUM)
	int photon = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2);
	#endif

	#if(DO_YE)
	int ye = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1);
	#endif
	#if(DONUCLEAR)
	int xalpha = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1);
	int xatm = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + 1);
	#endif
	#if(NEUTRINOS_M1)
	int uu_nu = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + read_nuclear * 2);
	int u1_nu = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + read_nuclear * 2 + 1);
	int u2_nu = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + read_nuclear * 2 + 2);
	int u3_nu = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + read_nuclear * 2 + 3);
	int number_nu = (8 + DOKTOT + read_M1 * 4 + read_Res * 3 + read_2T * 2 + read_Pnum * 1 + read_Ye * 1 + read_nuclear * 2 + 4);
	#endif

	//Set grid reduction factor
	if (BS1_read != BS_1 || BS2_read != BS_2 || BS3_read != BS_3) {
		red_1 = BS1_read / BS_1;
		red_2 = BS2_read / BS_2;
		red_3 = BS3_read / BS_3;
	}
	else red_1 = red_2 = red_3 = 1.0;

	ZSLOOP3D(-N1G + N1_GPU_offset[n] * red_1, (N1_GPU_offset[n] + BS_1) * red_1 - 1 + N1G, -N2G + N2_GPU_offset[n] * red_2, (N2_GPU_offset[n] + BS_2) * red_2 - 1 + N2G, -N3G + N3_GPU_offset[n] * red_3, (N3_GPU_offset[n] + BS_3) * red_3 - 1 + N3G) {
		for (k = 0; k < npr_file; k++) {
			fread(&(read[k]), double_size, 1, fp);
		}
		read_geom = 0;
		//Initialize variables
		if (i >= N1_GPU_offset[n] * red_1 && i <= (N1_GPU_offset[n] + BS_1) * red_1 && j >= N2_GPU_offset[n] * red_2 && j <= (N2_GPU_offset[n] + BS_2) * red_2 && z >= N3_GPU_offset[n] * red_3 && z <= (N3_GPU_offset[n] + BS_3) * red_3) {
			i1 = i / red_1;
			j1 = j / red_2;
			z1 = z / red_3;
			if ((i % red_1 == 0) && (j % red_2 == 0) && (z % red_3 == 0)) {
				for (k = 0; k < NPR; k++) p[nl[n]][index_3D(n, i1, j1, z1)][k] = 0.0;
				for (k = 0; k < NDIM; k++) ps[nl[n]][index_3D(n, i1, j1, z1)][k] = 0.0;
			}
		
			//Read in normal variables
			reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
			for (k = 0; k < NPR_U; k++) p[nl[n]][index_3D(n, i1, j1, z1)][k] += read[k] * reduce_factor;

			//Read in staggered grid
			#if(STAGGERED)
			reduce_factor = 1.0 / (double)(red_2 * red_3);
			if ((i % red_1 == 0))ps[nl[n]][index_3D(n, i1, j1, z1)][1] += read[npr_file - (NDIM - 1)] * reduce_factor / gdet[nl[n]][index_2D(n, i1, j1, z1)][FACE1];
			reduce_factor = 1.0 / (double)(red_1 * red_3);
			double fractheta_old = 1.e-2;
			if (N2 != 1) {
				fractheta_old = 1.0 - 2.0 / ((double)N2*red_2) * (BOUND_TYPE2==TRANSMISSIVE);
			}
			#if(SPHERICAL || SPHERICAL_GR)			
			if ((j % red_2 == 0))ps[nl[n]][index_3D(n, i1, j1, z1)][2] += read[npr_file - (NDIM - 2)] * reduce_factor / gdet[nl[n]][index_2D(n, i1, j1, z1)][FACE2] * fractheta / fractheta_old;
			#else
			if ((j % red_2 == 0))ps[nl[n]][index_3D(n, i1, j1, z1)][2] += read[npr_file - (NDIM - 2)] * reduce_factor / gdet[nl[n]][index_2D(n, i1, j1, z1)][FACE2];
			#endif
			reduce_factor = 1.0 / (double)(red_1 * red_2);
			if ((z % red_3 == 0))ps[nl[n]][index_3D(n, i1, j1, z1)][3] += read[npr_file - (NDIM - 3)] * reduce_factor / gdet[nl[n]][index_2D(n, i1, j1, z1)][FACE3];
			#endif

			//If file doesn't contain physics, initiliaze the physics just like in ICs
			#if(RAD_M1)
			if (!read_M1) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					init_rad_pres(p[nl[n]][index_3D(n, i1, j1, z1)]);
				}
				//dt = 1.e-5;
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][UU_RAD] += read[uu_rad] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][U1_RAD] += read[u1_rad] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][U2_RAD] += read[u2_rad] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][U3_RAD] += read[u3_rad] * reduce_factor;
			}
			#endif
			#if(TWO_T)
			if (!read_2T) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					double bsq;
					get_geometry(n, i1, j1, z1, CENT, &geom);
					read_geom = 1;
					bsq = bsq_calc(p[nl[n]][index_3D(n, i1, j1, z1)], &geom);
					set_2T_entropy(p[nl[n]][index_3D(n, i1, j1, z1)], bsq);
				}
			}
			else{
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][ENTRE] += read[entre] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][ENTRI] += read[entri] * reduce_factor;
			}
			#endif
			#if(P_NUM)
			if (!read_Pnum) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					double T_new, exp_xi, ucon[NDIM], ucon_rad[NDIM], ucov_rad[NDIM], u_dot_urad, urad_dot_urad, Ehat;
					if (!read_geom)get_geometry(n, i1, j1, z1, CENT, &geom);
					read_geom = 1;
					ucon_calc(p[nl[n]][index_3D(n, i1, j1, z1)], &geom, ucon);
					ucon_calc_rad(p[nl[n]][index_3D(n, i1, j1, z1)], &geom, ucon_rad);
					lower(ucon_rad, &geom, ucov_rad);
					u_dot_urad = ucon[0] * ucov_rad[0] + ucon[1] * ucov_rad[1] + ucon[2] * ucov_rad[2] + ucon[3] * ucov_rad[3];
					urad_dot_urad = ucon_rad[0] * ucov_rad[0] + ucon_rad[1] * ucov_rad[1] + ucon_rad[2] * ucov_rad[2] + ucon_rad[3] * ucov_rad[3];
					Ehat = ENERGY_DENSITY_SCALE * ((4. / 3.) * p[nl[n]][index_3D(n, i1, j1, z1)][UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * p[nl[n]][index_3D(n, i1, j1, z1)][UU_RAD] * (urad_dot_urad));
					T_new = pow(Ehat / ARAD, 0.25);
					p[nl[n]][index_3D(n, i1, j1, z1)][PHOTON] = p[nl[n]][index_3D(n, i1, j1, z1)][UU_RAD] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * T_new);
				}
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][PHOTON] += read[photon] * reduce_factor;
			}
			#endif
			#if(RESISTIVE)
			if (!read_Res) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					if (!read_geom)get_geometry(n, i1, j1, z1, CENT, &geom);
					set_E_init(p[nl[n]][index_3D(n, i1, j1, z1)], geom);
				}
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][E1] += read[e1] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][E2] += read[e2] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][E3] += read[e3] * reduce_factor;
			}
			#endif
			#if(DO_YE)
			if (!read_Ye) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					p[nl[n]][index_3D(n, i1, j1, z1)][YE] = 1.0;
				}
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][YE] += read[ye] * reduce_factor;
			}
			#endif
			#if(DONUCLEAR)
			if (!read_nuclear) {
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					p[nl[n]][index_3D(n, i1, j1, z1)][XALPHA] = 0.0;
					p[nl[n]][index_3D(n, i1, j1, z1)][XATM] = 1.0;
				}
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				p[nl[n]][index_3D(n, i1, j1, z1)][XALPHA] += read[xalpha] * reduce_factor;
				p[nl[n]][index_3D(n, i1, j1, z1)][XATM] += read[xatm] * reduce_factor;
			}
			#endif
			#if(NEUTRINOS_M1)
			int sp;
			if (!read_neutrinos) {
				double Tnu;
				if ((i % red_1) == (red_1 - 1) && (j % red_2) == (red_2 - 1) && (z % red_3) == (red_3 - 1)) {
					for (sp = 0; sp < NU_SPECIES; sp++) {
						p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(UU_NU, sp)] = 1e-30;
						p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U1_NU, sp)] = p[nl[n]][index_3D(n, i1, j1, z1)][U1];
						p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U2_NU, sp)] = p[nl[n]][index_3D(n, i1, j1, z1)][U2];
						p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U3_NU, sp)] = p[nl[n]][index_3D(n, i1, j1, z1)][U3];

						Tnu = pow(p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
						p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(NUMBER_NU, sp)] = p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tnu);
					}
				}
			}
			else {
				reduce_factor = 1.0 / (double)(red_1 * red_2 * red_3);
				for (sp = 0; sp < NU_SPECIES; sp++) {
					p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(UU_NU,sp)] += read[index_nu(uu_nu, sp)] * reduce_factor;
					p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U1_NU,sp)] += read[index_nu(u1_nu, sp)] * reduce_factor;
					p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U2_NU,sp)] += read[index_nu(u2_nu, sp)] * reduce_factor;
					p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(U3_NU,sp)] += read[index_nu(u3_nu, sp)] * reduce_factor;
					p[nl[n]][index_3D(n, i1, j1, z1)][index_nu(NUMBER_NU,sp)] += read[index_nu(number_nu,sp)] * reduce_factor;
				}
			}
			#endif
		}
	}
}

void add_toroidal_B(void) {
	int n, i, j, z, i0, j0, z0;
	double *rho_avg, *rho_avg_phi, *ug_avg, *ug_avg_phi, p_avg_phi, uu_avg_phi[NDIM], bsq_avg_phi, bsq_desired, r_avg_phi, *B3_avg_phi, X_avg_phi[NDIM], gcov_avg_phi[NDIM][NDIM], factor;
	double beta_desired = 4.0; //Target beta in thin disk

	//Allocate memory
	rho_avg = (double*)calloc((BS_1 * NB_1 + 2 * N1G) * (BS_2 * NB_2) * (BS_3 * NB_3), sizeof(double));
	rho_avg_phi = (double*)calloc((BS_1 * NB_1 + 2 * N1G) * (BS_2 * NB_2), sizeof(double));
	ug_avg = (double*)calloc((BS_1 * NB_1 + 2 * N1G) * (BS_2 * NB_2) * (BS_3 * NB_3), sizeof(double));
	ug_avg_phi = (double*)calloc((BS_1 * NB_1 + 2 * N1G) * (BS_2 * NB_2), sizeof(double));
	B3_avg_phi = (double*)calloc((BS_1 * NB_1 + 2 * N1G) * (BS_2 * NB_2), sizeof(double));

	//Calculate average ug and rho on 0th grid level
	for (n = 0; n < n_active; n++) {
		for (i = N1_GPU_offset[n_ord[n]]; i < N1_GPU_offset[n_ord[n]] + BS_1; i++)for (j = N2_GPU_offset[n_ord[n]]; j < N2_GPU_offset[n_ord[n]] + BS_2; j++)for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3; z++){
			i0 = i / pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]);
			j0 = j / pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]);
			z0 = z / pow(1 + REF_3, block[n_ord[n]][AMR_LEVEL3]);
			factor = (double)(pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) * pow(1 + REF_3, block[n_ord[n]][AMR_LEVEL3]));
			rho_avg[i0 * BS_3 * NB_3 * BS_2 * NB_2 + j0 * BS_3 * NB_3 + z0] += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / factor;
			ug_avg[i0 * BS_3 * NB_3 * BS_2 * NB_2 + j0 * BS_3 * NB_3 + z0] += (GAMMA - 1.0) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] / factor;
			#if(RAD_M1)
			ug_avg[i0 * BS_3 * NB_3 * BS_2 * NB_2 + j0 * BS_3 * NB_3 + z0] += (4.0 / 3.0 - 1.0) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD] / factor;
			#endif		
		}
	}

	//Calculate average in phi of all values
	for (i0 = 0; i0 < BS_1 * NB_1; i0++)for (j0 = 0; j0 < BS_2 * NB_2; j0++)for (z0 = 0; z0 < BS_3 * NB_3; z0++) {
		rho_avg_phi[i0 * BS_2 * NB_2 + j0] += rho_avg[i0 * BS_3 * NB_3 * BS_2 * NB_2 + j0 * BS_3 * NB_3 + z0] / (double)(BS_3 * NB_3);
		ug_avg_phi[i0 * BS_2 * NB_2 + j0] += ug_avg[i0 * BS_3 * NB_3 * BS_2 * NB_2 + j0 * BS_3 * NB_3 + z0] / (double)(BS_3 * NB_3);
	}
	MPI_Allreduce(MPI_IN_PLACE, rho_avg_phi, (BS_1 * NB_1) * (BS_2 * NB_2), MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, ug_avg_phi, (BS_1 * NB_1) * (BS_2 * NB_2), MPI_DOUBLE, MPI_SUM, mpi_cartcomm);

	//Set toroidal magnetic field at lowest AMR level
	for (i0 = 0; i0 < BS_1 * NB_1; i0++)for (j0 = 0; j0 < BS_2 * NB_2; j0++) {
		//Set coordinates and calculate the metric compoment
		X_avg_phi[1] = startx[1] + (i0 + 0.5) * (pow(log(Rout - RB), 1. / RADEXP) - pow(log(Rin - RB), 1. / RADEXP)) / (double)(N1);
		X_avg_phi[2] = startx[2] + (j0 + 0.5) * 2. * fractheta / (double)(N2);
		X_avg_phi[3] = 0.0;
		
		//Calculate radius
		r_avg_phi = exp(X_avg_phi[1]);

		if (rho_avg_phi[i0 * BS_2 * NB_2 + j0] > 0.000001 && r_avg_phi>-20.0) {
			//Calculate metric
			gcov_func(X_avg_phi, gcov_avg_phi);

			//Calculate present beta
			bsq_avg_phi = gcov_avg_phi[3][3]; //Rough approximation for bsq not taking into account the velocity of the fluid, which is <<c
			p_avg_phi = ug_avg_phi[i0 * BS_2 * NB_2 + j0];
			bsq_desired = 2.0 * p_avg_phi / beta_desired;

			//Set normalized B strengths
			B3_avg_phi[i0 * BS_2 * NB_2 + j0] = sqrt(bsq_desired / bsq_avg_phi);
		}
	}

	//Insert toroidal field into disk
	for (n = 0; n < n_active; n++) {
		for (i = N1_GPU_offset[n_ord[n]]; i < N1_GPU_offset[n_ord[n]] + BS_1; i++)for (j = N2_GPU_offset[n_ord[n]]; j < N2_GPU_offset[n_ord[n]] + BS_2; j++)for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3 + D3; z++) {
			i0 = i / pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]);
			j0 = j / pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] += B3_avg_phi[i0 * BS_2 * NB_2 + j0];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] += B3_avg_phi[i0 * BS_2 * NB_2 + j0];
		}
	}

	//Free memory
	free(rho_avg); 
	free(rho_avg_phi);
	free(ug_avg); 
	free(ug_avg_phi);
	free(B3_avg_phi);

	return;
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
		//if (rank == 1 % numtasks) {
		//	MPI_Wait(&req_rdumpgrid[0], &Statbound[nl[n_ord[0]]][0]);
		//	MPI_File_close(&grid_restart[0]);
		//}

		if (rank == 0 && fparam_restart != NULL)fclose(fparam_restart);

		//Now tell the writing is complete
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			if ((rdump_cnt - 1) % 2 == 0) checkfile = fopen("rdumps0/checkfile", "wb");
			else checkfile = fopen("rdumps1/checkfile", "wb");
			fwrite(&one, int_size, 1, checkfile);
			fclose(checkfile);
		}
	}
	first_rdump = 0;
}


int restart_read_param(void)
{
	int n, k;
	char filename[100], dirpath[100];
	FILE *param, *grid, *checkfile;
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
			sprintf(filename, "rdumps0/grid");
			param = fopen(filename, "rb");
			if (param != NULL) {
				gdump_grid_read(param);
				fclose(param);
			}
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
				if (rank == 0) fprintf(stderr, "Reading in rdumps1! \n");
				sprintf(filename, "rdumps1/grid");
				param = fopen(filename, "rb");
				if (param != NULL) {
					gdump_grid_read(param);
					fclose(param);
				}
				restart_number = 1;
			}
		}
	}

	if (t0 > t1 && value0==1) {
		sprintf(filename, "rdumps0/parameter");
		if (rank == 0) fprintf(stderr, "Reading in rdumps0! \n");
		param = fopen(filename, "rb");
		if (param != NULL) {
			param_read(param);
			fclose(param);
			sprintf(filename, "rdumps0/grid");
			param = fopen(filename, "rb");
			if (param != NULL) {
				gdump_grid_read(param);
				fclose(param);
			}
			restart_number = 0;
		}
	}

	#if(CALC_MDOT)
	set_mass_density_scale(&mass_density_scale_cpu, &magnetic_density_scale_cpu);
	#endif

	#if(CALC_METRIC)
	set_metric_scale();
	#endif

	if (restart_number == -1) {
		if(rank==0) fprintf(stderr, "No restart dump available! \n");
		return 0;
	}
	else {
		return 1;
	}
}

void param_read(FILE *fp) {
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	int u, n, n2;
	int exit_r = 0;
	double dummy;
	u = rdump_cnt + 1;

	//Read in essential stuff for restart
	fread(&t, double_size, 1, fp);
	fread(&n_active, int_size, 1, fp);
	fread(&n_active_total, int_size, 1, fp);
	fread(&nstep, int_size, 1, fp);
	fread(&DTd, double_size, 1, fp);
	fread(&DTl, double_size, 1, fp);
	fread(&dummy, double_size, 1, fp);
	fread(&dump_cnt, int_size, 1, fp);
	fread(&u, int_size, 1, fp);
	fread(&dt, double_size, 1, fp);
	fread(&failed, int_size, 1, fp);

	//Read in stuff that should be checked later
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
	double Rin_read;
	double Rout_read;
	double mdot_read;
	double gam_read;
	double a_read;
	double cour_read;
	double startx_read[NDIM];
	double dx_read[NDIM];

	fread(&BS1_read, int_size, 1, fp);
	fread(&BS2_read, int_size, 1, fp);
	fread(&BS3_read, int_size, 1, fp);
	fread(&NB_print, int_size, 1, fp);
	fread(&NB1_print, int_size, 1, fp);
	fread(&NB2_print, int_size, 1, fp);
	fread(&NB3_print, int_size, 1, fp);
	fread(&startx_read[1], double_size, 1, fp);
	fread(&startx_read[2], double_size, 1, fp);
	fread(&startx_read[3], double_size, 1, fp);
	fread(&dx_read[1], double_size, 1, fp);
	fread(&dx_read[2], double_size, 1, fp);
	fread(&dx_read[3], double_size, 1, fp);
	fread(&tf, double_size, 1, fp);
	fread(&a_read, double_size, 1, fp);
	fread(&gam_read, double_size, 1, fp);
	fread(&cour_read, double_size, 1, fp);
	fread(&Rin_read, double_size, 1, fp);
	fread(&Rout_read, double_size, 1, fp);
	fread(&mdot_read, double_size, 1, fp);
	fread(&dummy, double_size, 1, fp);
	fread(&lim, int_size, 1, fp);
	fread(&stag, int_size, 1, fp);
	fread(&dump_cnt_reduced, int_size, 1, fp);
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

	if (docyl >= 1000000) {
		read_nuclear = 1;
		docyl -= 1000000;
	}
	else read_nuclear = 0;

	if (docyl >= 100000) {
		read_neutrinos = 1;
		docyl -= 100000;
	}
	else read_neutrinos = 0;
	
	if (docyl >= 10000) {
		read_Ye = 1;
		docyl -= 10000;
	}
	else read_Ye = 0;

	if (docyl >= 1000) {
		read_Pnum = 1;
		docyl -= 1000;
	}
	else read_Pnum = 0;
	
	if (docyl >= 100) {
		read_2T = 1;
		docyl -= 100;
	}
	else read_2T = 0;
	
	if (docyl >= 10) {
		read_Res = 1;
		docyl -= 10;
	}
	else read_Res = 0;
	
	if (docyl >= 1) {
		read_M1 = 1;
		docyl -= 1;
	}
	else {
		read_M1 = 0;
		read_M1_2 = 1;
	}
	
	fread(&dk, int_size, 1, fp);

	//First deactivate all blocks
	for (n = 0; n < NB; n++) {
		block[n][AMR_ACTIVE] = 0;
		block[n][AMR_NODE] = -1;
		block[n][AMR_TIMELEVEL] = 1;
	}

	//Now check which blocks are active in grid
	for (n = 0; n < n_active_total; n++) {
		fread(&n2, int_size, 1, fp);
		if (n2 > NB - 1)fprintf(stderr, "Param read error 1. \n");

		block[n2][AMR_ACTIVE] = 1;
		fread(&block[n2][AMR_TIMELEVEL], int_size, 1, fp);
		block[n2][AMR_TIMELEVEL] = MY_MIN(block[n2][AMR_TIMELEVEL], AMR_MAXTIMELEVEL);
		fread(&block[n2][AMR_NODE], int_size, 1, fp);
		block[n2][AMR_NODE] = -1;
	}

	if ( NB1_print != NB_1 || NB2_print != NB_2 || NB3_print != NB_3 || a!=BH_SPIN) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. Your code will probably segfault. Make sure the restart file is compatible with the present code and grid parameters! \n");
		}
		exit_r = 1;
	}
	if (stag != STAGGERED) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. Staggered grid not set properly! \n");
		}
		exit_r = 1;
	}
	if (Rout_read != Rout) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. Rout not set properly! \n");
		}
		exit_r = 1;
	}
	if (Rin_read != Rin) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. Rin not set properly! \n");
		}
		exit_r = 1;
	}
	#if(CALC_MDOT)
	if (mdot_read < 1e-6) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. Mdot suspiciously low! \n");
		}
		exit_r = 1;
	}
	#endif
	if (a_read != a) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. a not set properly! \n");
		}
		exit_r = 1;
	}
	if (gam_read != gam) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. gam not set properly! \n");
		}
		exit_r = 1;
	}
	if (startx_read[1] != startx[1] || startx_read[2] != startx[2] || startx_read[3] != startx[3]) {
		if (rank == 0) {
			fprintf(stderr, "Error reading in input parameters. startx not set properly! \n");
		}
		//exit_r = 1;
	}
	if (cour_read != cour) {
		if (rank == 0) {
			fprintf(stderr, "Warning reading in input parameters. Changin courant factor from %f to %f! \n", cour_read, cour);
		}
	}
	if (BS1_read != BS_1 || BS2_read != BS_2 || BS3_read != BS_3) {
		if (BS1_read % BS_1 == 0 && BS2_read % BS_2 == 0 && BS3_read % BS_3 == 0) {
			if (rank == 0) fprintf(stderr, "Downscaling bigger data set of original resolution of %dx%dx%d to resolution %dx%dx%d! \n", BS1_read, BS2_read, BS3_read, BS_1, BS_2, BS_3);
		}
		else {
			if (rank == 0) fprintf(stderr, "Error reading in input parameters. Failed upscaling resolution due to incompatible ratios! \n");
			exit_r = 1;
		}
	}

	if (exit_r) {
		//exit(0);
	}

	//Set mass density scale
	#if(CALC_MDOT)
	mdot_cpu = mdot_read;
	#endif

	//Set nstep to 0 for convenience
	nstep = 0;
}

//Calculate Mdot
double calc_Mdot() {
	int n, i, j, z, k, icalc;
	double rcalc = 5.0; //Radius at which to calculate mdot
	double mdot = 0.;
	struct of_geom geom;
	struct of_state q;

	//Calculate Mdot at r=rcalc
	for (n = 0; n < n_active; n++) {
		//Set index at which to calculate mdot
		icalc = (int)((log(rcalc) - log(Rin))) / dx[nl[n_ord[n]]][1];

		//Loop over cells in theta-phi plane
		if ((icalc >= N1_GPU_offset[n_ord[n]]) && (icalc < N1_GPU_offset[n_ord[n]] + BS_1)) {
			#if(GPU_ENABLED)
			gpuMemcpyAsync(p_1[nl[n]], Bufferp_1[nl[n]], (int)(5 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]])) * sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
			gpuDeviceSynchronize();
			#pragma omp parallel private(i, j, z, k)
			{
				#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
				ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
					for (k = 0; k < 5; k++) {
						p[nl[n]][index_3D(n, i, j, z)][k] = p_1[nl[n]][k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
					}
				}
			}
			#endif

			ZSLOOP3D(icalc, icalc, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
				get_geometry(n_ord[n], i, j, z, CENT, &geom);
				get_state(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom, &q);

				mdot += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * q.ucon[1] * geom.g * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
		}
	}
	//Sum over MPI processes
	MPI_Allreduce(MPI_IN_PLACE, &mdot, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);

	//Return absolute value of mdot
	return fabs(mdot);
}

//Calculate mass density scaling
void set_mass_density_scale(double* mass_density_scale_cpu, double* magnetic_density_scale_cpu) {
	double mdot_target, mdot_cgs, mdot_cgs_edd, scaling_factor;
	double L_dot_edd, M_dot_edd, efficiency;
	double n_steps;

	//Initialize mdot_cpu and t_mdot at start of run
	if (!isfinite(mdot_cpu)) mdot_cpu = 0.;
	if (!isfinite(t_mdot)) t_mdot = t-1.0e-5;
	if (!isfinite(magnetic_density_scale_cpu[0])) magnetic_density_scale_cpu[0] = 1.0;

	//Check input
	if (0.1 * T_DOUBLE / T_MDOT < 10) {
		if (rank == 0)fprintf(stderr, "Error in set_mass_density_scale! \n");
		exit(0);
	}

	//Calculate updated mdot
	if (t >= t_mdot) {
		t_mdot += T_MDOT;

		n_steps = 0.1 * T_DOUBLE / T_MDOT;
		mdot_cpu = (1.0 - 1.0 / n_steps) * mdot_cpu + (1.0 / n_steps) * calc_Mdot();

		//Convert mdot_cpu to Eddington units
		mdot_cgs = mdot_cpu * MASS_DENSITY_SCALE * (R_G_CGS * R_G_CGS * R_G_CGS) / R_GOC_CGS;
		L_dot_edd = 1.3 * pow(10.0, 46.0) * M_SGRA_SOLAR / pow(10.0, 8.0);
		efficiency = 0.178; //For a=0.9375 BH
		M_dot_edd = (1.0 / efficiency) * L_dot_edd / (C_CGS * C_CGS);
		mdot_cgs_edd = mdot_cgs / M_dot_edd;

		//Set equal to mass density scale before T_INIT
		if (t < T_INIT) {
			mass_density_scale_cpu[0] = MASS_DENSITY_SCALE;
			magnetic_density_scale_cpu[0] = 1.0;
		}
		else {
			//Flip sign for super-Eddington flow
			#if(HIGH_MDOT)
			mdot_target = MDOT_START * pow(2.0, -(t - T_INIT) / T_DOUBLE);
			#else
			mdot_target = MDOT_START * pow(2.0, (t - T_INIT) / T_DOUBLE);
			#endif
			scaling_factor = mdot_target / mdot_cgs_edd;
			mass_density_scale_cpu[0] = scaling_factor * MASS_DENSITY_SCALE;
			magnetic_density_scale_cpu[0] = 1.0;// pow(2.0, -(t - T_INIT) / T_DOUBLE);
		}
	}
}

//Calculate mass of black hole as function of time
double calc_MBH(void) {
	int n, i, j, z, k, icalc;
	double rcalc = 5.0; //Radius at which to calculate mdot
	double mdot = 0.;
	struct of_geom geom;
	struct of_state q;

	//Calculate Mdot at r=rcalc
	for (n = 0; n < n_active; n++) {
		//Set index at which to calculate mdot
		icalc = (int)((log(rcalc) - log(Rin))) / dx[nl[n_ord[n]]][1];

		//Loop over cells in theta-phi plane
		if ((icalc > N1_GPU_offset[n_ord[n]]) && (icalc < N1_GPU_offset[n_ord[n]] + BS_1)) {
			#if(GPU_ENABLED)
			gpuMemcpyAsync(p_1[nl[n]], Bufferp_1[nl[n]], (int)(5 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]])) * sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
			gpuDeviceSynchronize();

			#pragma omp parallel private(i, j, z, k)
			{
				#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
				ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
					for (k = 0; k < 5; k++) {
						p[nl[n]][index_3D(n, i, j, z)][k] = p_1[nl[n]][k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
					}
				}
			}
			#endif

			ZSLOOP3D(icalc, icalc, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
				get_geometry(n_ord[n], i, j, z, CENT, &geom);
				get_state(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom, &q);

				mdot += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * q.ucon[1] * geom.g * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
		}
	}

	//Sum over MPI processes
	MPI_Allreduce(MPI_IN_PLACE, &mdot, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	
	//Reset at start of run total accreted mass
	if (!isfinite(accreted_mass))accreted_mass = 0;
	if (!isfinite(t_prev)) t_prev = 0.0;

	//Add mass accreted to accreted_mass variable
	accreted_mass += (mdot * (t - t_prev) * MASS_DENSITY_SCALE * R_G_CGS * R_G_CGS * R_G_CGS); //ORE: Need to check this

	//Set t_prev to current time
	t_prev = t;

	//Return total mass of black
	return fabs(1.0 + accreted_mass / (M_SGRA_SOLAR * M_SOLAR_CGS));
}

//Sets density scale of metric as function of time
void set_metric_scale(void) {
	double MBH;
	int n;

	//Calculate mass of black hole
	if (!isfinite(metric_scale_cpu))metric_scale_cpu = 1.0;
	MBH = calc_MBH();

	//Calculate Mdot
	if (t > 0) metric_scale_cpu = MBH;
	else metric_scale_cpu = 1.0;

	//Recalculate metric when necessary
	for (n = 0; n < n_active; n++) {
		recalculate_metric(n_ord[n]);
	}
}

//Recalculate metric for increased black hole mass
void recalculate_metric(int n) {
	int i, j, z, k;

	//Copy data from GPU to CPU
	#if(GPU_ENABLED || GPU_DEBUG )
	gpuMemcpyAsync(ps_1[nl[n]], Bufferps_1[nl[n]], (int)(3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]])) * sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	gpuDeviceSynchronize();

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
			for (k = 1; k < NDIM; k++) {
				ps[nl[n]][index_3D(n, i, j, z)][k] = ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
		}
	}
	#endif

	//Normalize staggered magnetic field components
	ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		psh[nl[n]][index_3D(n, i, j, z)][1] = gdet[nl[n]][index_2D(n, i, j, z)][FACE1] * ps[nl[n]][index_3D(n, i, j, z)][1];
		psh[nl[n]][index_3D(n, i, j, z)][2] = gdet[nl[n]][index_2D(n, i, j, z)][FACE2] * ps[nl[n]][index_3D(n, i, j, z)][2];
		psh[nl[n]][index_3D(n, i, j, z)][3] = gdet[nl[n]][index_2D(n, i, j, z)][FACE3] * ps[nl[n]][index_3D(n, i, j, z)][3];
	}

	//Recalculate metric and connection coefficients
	set_grid(n);

	//De-normalize staggered magnetic field components
	ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		ps[nl[n]][index_3D(n, i, j, z)][1] = psh[nl[n]][index_3D(n, i, j, z)][1] / gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
		ps[nl[n]][index_3D(n, i, j, z)][2] = psh[nl[n]][index_3D(n, i, j, z)][2] / gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
		ps[nl[n]][index_3D(n, i, j, z)][3] = psh[nl[n]][index_3D(n, i, j, z)][3] / gdet[nl[n]][index_2D(n, i, j, z)][FACE3];
	}

	//Reset half step variables
	ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
		psh[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, j, z)][1];
		psh[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i, j, z)][2];
		psh[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i, j, z)][3];
	}

	//Copy data back to GPU
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_write_metric(n); //Matthew: Can be done faster
	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++) {
				ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[nl[n]][index_3D(n, i, j, z)][k];
			}
			#endif
		}
	}
		#if(STAGGERED)
		gpuMemcpyAsync(Bufferps_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
		gpuMemcpyAsync(Bufferpsh_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
		#endif
	#endif
}
