#include "decs_MPI.h"

void pack_send_B1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					send[n][(i - i1)*zsize*jsize + (j - j1)*zsize + (z - z1)] = prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
		}
	}
}

void pack_send_B2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					send[n][(j - j1)*zsize*isize + (i - i1)*zsize + (z - z1)] = prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
		}
	}
}

void pack_send_B3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					send[n][(z - z1)*jsize*isize + (i - i1)*jsize + (j - j1)] = prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
		}
	}
}

void pack_send_B_average1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j += 1 + REF_2){
				for (z = z1; z < z2; z += (1 + REF_3)){
					send[n][(i - i1) *zsize*jsize + (j - j1) / (1 + REF_2)*zsize + (z - z1) / (1 + REF_3)]
						= 0.25*(F1[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						F1[n][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						F1[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][1] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][FACE1]
						+ F1[n][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][1] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][FACE1]);
				}
			}
		}
	}
}

void pack_send_B_average2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict F2[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (z = z1; z < z2; z += 1 + REF_3){
					send[n][(j - j1)*isize*zsize + (i - i1) / (1 + REF_1)*zsize + (z - z1) / (1 + REF_3)]
						= 0.25*(F2[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						F2[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][2] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][FACE2] +
						F2[n][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index_2D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ F2[n][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][2] * gdet[n][index_2D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][FACE2]);
				}
			}
		}
	}
}

void pack_send_B_average3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict F3[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z)
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (j = j1; j < j2; j += 1 + REF_2){
					send[n][(z - z1)*isize*jsize + (i - i1) / (1 + REF_1)*jsize + (j - j1) / (1 + REF_2)]
						= 0.25*(F3[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						F3[n][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index_2D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						F3[n][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index_2D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ F3[n][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index_2D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]);
				}
			}
		}
	}
}


void unpack_receive_B1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,factor)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					factor = 1.;
					if (div == 1) factor = gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						= receive[n_rec][(i - i1)*zsize*jsize + (j - j1)*zsize + (z - z1)] / factor;
				}
			}
		}
	}
}

void unpack_receive_B2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent, int neg){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,factor)
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					factor = 1;
					if (neg == 1)factor = -1.;
					if (neg==0 && div == 1) factor = gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					else if (neg == 1 && div == 1) factor = -gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						= receive[n_rec][(j - j1)*zsize*isize + (i - i1)*zsize + (z - z1)] / factor;
				}
			}
		}
	}
}

void unpack_receive_B3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,factor)
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					factor = 1;
					if (div == 1) factor = gdet[n][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					prim[n][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						= receive[n_rec][(z - z1)*isize*jsize + (i - i1)*jsize + (j - j1)] / factor;
				}
			}
		}
	}
}

/*Send boundaries between compute nodes through MPI*/
void B_send1(double(*restrict F1[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive4_fine[n][0], NDIM * (N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (440 * NB + block[n][AMR_NBR2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][440]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive4_5fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][545]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive4_6fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][546]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive4_7fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][547]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive4_8fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][548]);
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average1(n, N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_fine, F1, &(Bufferp[n]), &(Buffersend2fine[n]), &(boundevent[n][520]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						
					}
					rc += MPI_Isend(&send2_fine[n][0], NDIM*(N3_GPU[n]) / (1 + REF_3)*(N2_GPU[n]) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (520 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1){
			pack_send_B1(n, 0, 1, 0, N2_GPU[n], 0, N3_GPU[n],
				N2_GPU[n], N3_GPU[n], send4_fine, F1, &(Bufferp[n]), &(Buffersend4fine[n]), &(boundevent[n][540]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
				
				}
				rc += MPI_Isend(&send4_fine[n][0], NDIM*N3_GPU[n] * N2_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (440 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[n]);
				MPI_Request_free(&req[n]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive2_1fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][521]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive2_2fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][522]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive2_3fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][523]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]&& REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive2_4fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][524]);
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average1(n, 0, 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_fine, F1, &(Bufferp[n]), &(Buffersend4fine[n]), &(boundevent[n][540]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						
					}
					rc += MPI_Isend(&send4_fine[n][0], NDIM*(N3_GPU[n]) / (1 + REF_3)*(N2_GPU[n]) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (540 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
#endif
}

void B_send2(double(*restrict F2[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] < NB_3*pow(1 + REF_3, block[n][AMR_LEVEL]) / 2){
					pack_send_B2(n, 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], send3_fine, F2, &(Bufferp[n]), &(Buffersend3fine[n]), &(boundevent[n][530]));
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
						
						}
						rc += MPI_Isend(&send3_fine[n][0], NDIM*N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (430 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[n]);
						MPI_Request_free(&req[n]);
					}
				}
				else if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL]) / 2){
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Irecv(&receive1_fine[n][0], NDIM * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (430 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][410]);
					}
				}
			}
			else{
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Irecv(&receive1_fine[n][0], NDIM * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (410 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][410]);
				}
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive1_3fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][513]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive1_4fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][514]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]&& REF_1 == 1){
			rc += MPI_Irecv(&receive1_7fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][517]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive1_8fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][518]);
		}
		
		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average2(n, 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_fine, F2, &(Bufferp[n]), &(Buffersend3fine[n]), &(boundevent[n][530]));
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
					
					}
					rc += MPI_Isend(&send3_fine[n][0], NDIM*(N3_GPU[n]) / (1 + REF_3)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (530 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){	
			if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] < NB_3*pow(1 + REF_3, block[n][AMR_LEVEL]) / 2){
					pack_send_B2(n, 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], send1_fine, F2, &(Bufferp[n]), &(Buffersend1fine[n]), &(boundevent[n][510]));
					if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
							
						}
						rc += MPI_Isend(&send1_fine[n][0], NDIM*N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (410 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[n]);
						MPI_Request_free(&req[n]);
					}
				}
				else if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL]) / 2){
					if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Irecv(&receive3_fine[n][0], NDIM * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (410 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][430]);
					}
				}
			}
			else{
				pack_send_B2(n, 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], send1_fine, F2, &(Bufferp[n]), &(Buffersend1fine[n]), &(boundevent[n][510]));
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						
					}
					rc += MPI_Isend(&send1_fine[n][0], NDIM*N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (410 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive3_1fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][531]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1 ){
			rc += MPI_Irecv(&receive3_2fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][532]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 ){
			rc += MPI_Irecv(&receive3_5fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][535]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive3_6fine[n][0], NDIM*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][536]);
		}
		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average2(n, 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_fine, F2, &(Bufferp[n]), &(Buffersend1fine[n]), &(boundevent[n][510]));
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
					
					}
					rc += MPI_Isend(&send1_fine[n][0], NDIM*(N3_GPU[n]) / (1 + REF_3)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (510 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
#endif
}

void B_send3(double(*restrict F3[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive6_fine[n][0], NDIM * N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (460 * NB + block[n][AMR_NBR5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][460]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive6_2fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][562]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive6_4fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][564]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive6_6fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][566]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive6_8fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][568]);
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average3(n, 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_fine, F3, &(Bufferp[n]), &(Buffersend5fine[n]), &(boundevent[n][550]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						
					}
					rc += MPI_Isend(&send5_fine[n][0], NDIM*(N2_GPU[n]) / (1 + REF_2)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (550 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1){
			pack_send_B3(n, 0, N1_GPU[n], 0, N2_GPU[n], 0, D3, N1_GPU[n], N2_GPU[n], send6_fine, F3, &(Bufferp[n]), &(Buffersend6fine[n]), &(boundevent[n][560]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
				
				}
				rc += MPI_Isend(&send6_fine[n][0], NDIM*N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (460 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[n]);
				MPI_Request_free(&req[n]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive5_1fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][551]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive5_3fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][553]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive5_5fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][555]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive5_7fine[n][0], NDIM*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][557]);
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B_average3(n, 0, N1_GPU[n], 0, N2_GPU[n], 0, D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_fine, F3, &(Bufferp[n]), &(Buffersend6fine[n]), &(boundevent[n][560]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						
					}
					rc += MPI_Isend(&send6_fine[n][0], NDIM*(N2_GPU[n]) / (1 + REF_2)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (560 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}


/*Receive boundaries for compute nodes through MPI*/
void B_rec1(double(*restrict F1[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][521], &Statbound[n][521]);
				unpack_receive_B1(n, n, 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_1fine, F1, 1, &(Bufferp[n]), &(Bufferrec2_1fine[n]), NULL);
			}
			else{
				unpack_receive_B1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_fine, F1, 1, &(Bufferp[n]), &(Buffersend2fine[block[block[n][AMR_NBR4]][AMR_CHILD5]]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD5]][520]));
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][522], &Statbound[n][522]);
					unpack_receive_B1(n, n, 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_2fine, F1, 1, &(Bufferp[n]), &(Bufferrec2_2fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_fine, F1, 1, &(Bufferp[n]), &(Buffersend2fine[block[block[n][AMR_NBR4]][AMR_CHILD6]]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD6]][520]));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][523], &Statbound[n][523]);
					unpack_receive_B1(n, n, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_3fine, F1, 1, &(Bufferp[n]), &(Bufferrec2_3fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_fine, F1, 1, &(Bufferp[n]), &(Buffersend2fine[block[block[n][AMR_NBR4]][AMR_CHILD7]]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD7]][520]));
				}
			}
			if (REF_2 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][524], &Statbound[n][524]);
					unpack_receive_B1(n, n, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_4fine, F1, 1, &(Bufferp[n]), &(Bufferrec2_4fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_fine, F1, 1, &(Bufferp[n]), &(Buffersend2fine[block[block[n][AMR_NBR4]][AMR_CHILD8]]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD8]][520]));
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][440], &Statbound[n][440]);
				unpack_receive_B1(n, n, N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n], receive4_fine, F1, 0, &(Bufferp[n]), &(Bufferrec4fine[n]), NULL);
			}
			else{
				unpack_receive_B1(n, block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n], send4_fine, F1, 0, &(Bufferp[n]), &(Buffersend4fine[block[n][AMR_NBR2]]), &(boundevent[block[n][AMR_NBR2]][540]));
			}		
		}	
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][545], &Statbound[n][545]);
				unpack_receive_B1(n, n, N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_5fine, F1, 1, &(Bufferp[n]), &(Bufferrec4_5fine[n]), NULL);
			}
			else{
				unpack_receive_B1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_fine, F1, 1, &(Bufferp[n]), &(Buffersend4fine[block[block[n][AMR_NBR2]][AMR_CHILD1]]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD1]][540]));
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][546], &Statbound[n][546]);
					unpack_receive_B1(n, n, N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_6fine, F1, 1, &(Bufferp[n]), &(Bufferrec4_6fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_fine, F1, 1, &(Bufferp[n]), &(Buffersend4fine[block[block[n][AMR_NBR2]][AMR_CHILD2]]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD2]][540]));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][547], &Statbound[n][547]);
					unpack_receive_B1(n, n, N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_7fine, F1, 1, &(Bufferp[n]), &(Bufferrec4_7fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_fine, F1, 1, &(Bufferp[n]), &(Buffersend4fine[block[block[n][AMR_NBR2]][AMR_CHILD3]]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD3]][540]));
				}
			}
			if (REF_2==1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][548], &Statbound[n][548]);
					unpack_receive_B1(n, n, N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_8fine, F1, 1, &(Bufferp[n]), &(Bufferrec4_8fine[n]), NULL);
				}
				else{
					unpack_receive_B1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_fine, F1, 1, &(Bufferp[n]), &(Buffersend4fine[block[block[n][AMR_NBR2]][AMR_CHILD4]]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD4]][540]));
				}
			}
		}
	}
#endif
}

void B_rec2(double(*restrict F2[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && block[n][AMR_COORD3] >=NB_3*pow(1+REF_3,block[n][AMR_LEVEL])/2){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][430], &Statbound[n][430]);
					unpack_receive_B2(n, n, 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], receive3_fine, F2, 0, &(Bufferp[n]), &(Bufferrec3fine[n]), NULL, 1);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], send1_fine, F2, 0, &(Bufferp[n]), &(Buffersend1fine[block[n][AMR_NBR1]]), &(boundevent[block[n][AMR_NBR1]][510]), 1);
				}
			}
			else{
			
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][531], &Statbound[n][531]);
				unpack_receive_B2(n, n, 0, N1_GPU[n] / (1 + REF_1), 0, 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_1fine, F2, 1, &(Bufferp[n]), &(Bufferrec3_1fine[n]), NULL, 0);
			}
			else{
				unpack_receive_B2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), 0, 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_fine, F2, 1, &(Bufferp[n]), &(Buffersend3fine[block[block[n][AMR_NBR1]][AMR_CHILD3]]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD3]][530]), 0);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][532], &Statbound[n][532]);
					unpack_receive_B2(n, n, 0, N1_GPU[n] / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_2fine, F2, 1, &(Bufferp[n]), &(Bufferrec3_2fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_fine, F2, 1, &(Bufferp[n]), &(Buffersend3fine[block[block[n][AMR_NBR1]][AMR_CHILD4]]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD4]][530]), 0);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][535], &Statbound[n][535]);
					unpack_receive_B2(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_5fine, F2, 1, &(Bufferp[n]), &(Bufferrec3_5fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_fine, F2, 1, &(Bufferp[n]), &(Buffersend3fine[block[block[n][AMR_NBR1]][AMR_CHILD7]]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD7]][530]), 0);
				}
			}
			if (REF_1==1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][536], &Statbound[n][536]);
					unpack_receive_B2(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_6fine, F2, 1, &(Bufferp[n]), &(Bufferrec3_6fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_fine, F2, 1, &(Bufferp[n]), &(Buffersend3fine[block[block[n][AMR_NBR1]][AMR_CHILD8]]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD8]][530]), 0);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL]) / 2){
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						MPI_Wait(&boundreqs[n][410], &Statbound[n][410]);
						unpack_receive_B2(n, n, 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
							N1_GPU[n], N3_GPU[n], receive1_fine, F2, 0, &(Bufferp[n]), &(Bufferrec1fine[n]), NULL, 1);
					}
					else{
						unpack_receive_B2(n, block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
							N1_GPU[n], N3_GPU[n], send3_fine, F2, 0, &(Bufferp[n]), &(Buffersend3fine[block[n][AMR_NBR3]]), &(boundevent[block[n][AMR_NBR3]][530]), 1);
					}
				}
			}
			else{
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][410], &Statbound[n][410]);
					unpack_receive_B2(n, n, 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], receive1_fine, F2, 0, &(Bufferp[n]), &(Bufferrec1fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
						N1_GPU[n], N3_GPU[n], send1_fine, F2, 0, &(Bufferp[n]), &(Buffersend1fine[block[n][AMR_NBR3]]), &(boundevent[block[n][AMR_NBR3]][510]), 0);
				}
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][513], &Statbound[n][513]);
				unpack_receive_B2(n, n, 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_3fine, F2, 1, &(Bufferp[n]), &(Bufferrec1_3fine[n]), NULL, 0);
			}
			else{
				unpack_receive_B2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_fine, F2, 1, &(Bufferp[n]), &(Buffersend1fine[block[block[n][AMR_NBR3]][AMR_CHILD1]]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD1]][510]), 0);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][514], &Statbound[n][514]);
					unpack_receive_B2(n, n, 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_4fine, F2, 1, &(Bufferp[n]), &(Bufferrec1_4fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_fine, F2, 1, &(Bufferp[n]), &(Buffersend1fine[block[block[n][AMR_NBR3]][AMR_CHILD2]]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD2]][510]), 0);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][517], &Statbound[n][517]);
					unpack_receive_B2(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_7fine, F2, 1, &(Bufferp[n]), &(Bufferrec1_7fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_fine, F2, 1, &(Bufferp[n]), &(Buffersend1fine[block[block[n][AMR_NBR3]][AMR_CHILD5]]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD5]][510]), 0);
				}
			}
			if (REF_1==1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][518], &Statbound[n][518]);
					unpack_receive_B2(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_8fine, F2, 1, &(Bufferp[n]), &(Bufferrec1_8fine[n]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_fine, F2, 1, &(Bufferp[n]), &(Buffersend1fine[block[block[n][AMR_NBR3]][AMR_CHILD6]]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD6]][510]), 0);
				}
			}
		}
	}
#endif
}
void B_rec3(double(*restrict F3[NB])[NDIM], double * Bufferp[NB], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][551], &Statbound[n][551]);
				unpack_receive_B3(n, n, 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), 0, D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_1fine, F3, 1, &(Bufferp[n]), &(Bufferrec5_1fine[n]), NULL);
			}
			else{
				unpack_receive_B3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), 0, D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_fine, F3, 1, &(Bufferp[n]), &(Buffersend5fine[block[block[n][AMR_NBR6]][AMR_CHILD2]]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD2]][550]));
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][553], &Statbound[n][553]);
					unpack_receive_B3(n, n, 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_3fine, F3, 1, &(Bufferp[n]), &(Bufferrec5_3fine[n]), NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_fine, F3, 1, &(Bufferp[n]), &(Buffersend5fine[block[block[n][AMR_NBR6]][AMR_CHILD4]]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD4]][550]));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][555], &Statbound[n][555]);
					unpack_receive_B3(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_5fine, F3, 1, &(Bufferp[n]), &(Bufferrec5_5fine[n]), NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_fine, F3, 1, &(Bufferp[n]), &(Buffersend5fine[block[block[n][AMR_NBR6]][AMR_CHILD6]]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD6]][550]));
				}
			}
			if (REF_1==1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][557], &Statbound[n][557]);
					unpack_receive_B3(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_7fine, F3, 1, &(Bufferp[n]), &(Bufferrec5_7fine[n]), NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_fine, F3, 1, &(Bufferp[n]), &(Buffersend5fine[block[block[n][AMR_NBR6]][AMR_CHILD8]]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD8]][550]));
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][460], &Statbound[n][460]);
				unpack_receive_B3(n, n, 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n], N2_GPU[n], receive6_fine, F3, 0, &(Bufferp[n]), &(Bufferrec6fine[n]), NULL);
			}
			else{
				unpack_receive_B3(n, block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n], N2_GPU[n], send6_fine, F3, 0, &(Bufferp[n]), &(Buffersend6fine[block[n][AMR_NBR5]]), &(boundevent[block[n][AMR_NBR5]][560]));
			}
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[n][562], &Statbound[n][562]);
				unpack_receive_B3(n, n, 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_2fine, F3, 1, &(Bufferp[n]), &(Bufferrec6_2fine[n]), NULL);
			}
			else{
				unpack_receive_B3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_fine, F3, 1, &(Bufferp[n]), &(Buffersend6fine[block[block[n][AMR_NBR5]][AMR_CHILD1]]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD1]][560]));
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][564], &Statbound[n][564]);
					unpack_receive_B3(n, n, 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_4fine, F3, 1, &(Bufferp[n]), &(Bufferrec6_4fine[n]), NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_fine, F3, 1, &(Bufferp[n]), &(Buffersend6fine[block[block[n][AMR_NBR5]][AMR_CHILD3]]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD3]][560]));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][566], &Statbound[n][566]);
					unpack_receive_B3(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_6fine, F3, 1, &(Bufferp[n]), &(Bufferrec6_6fine[n]),NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_fine, F3, 1, &(Bufferp[n]), &(Buffersend6fine[block[block[n][AMR_NBR5]][AMR_CHILD5]]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD5]][560]));
				}
			}
			if (REF_1==1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[n][568], &Statbound[n][568]);
					unpack_receive_B3(n, n, N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_8fine, F3, 1, &(Bufferp[n]), &(Bufferrec6_8fine[n]), NULL);
				}
				else{
					unpack_receive_B3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_fine, F3, 1, &(Bufferp[n]), &(Buffersend6fine[block[block[n][AMR_NBR5]][AMR_CHILD7]]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD7]][560]));
				}
			}
		}
	}
		//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Send boundaries between compute nodes through MPI*/
void Bp_send1(double(*restrict F1[NB])[NDIM], int n){
#if (MPI_enable)
	int i;
	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive4_5[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][0]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive4_6[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][1]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive4_7[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][2]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive4_8[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (540 * NB + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][3]);
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B1(n, N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n], send2, F1, NULL, NULL, NULL);
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send2[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]) , MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (520 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive2_1[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][4]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive2_2[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][5]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive2_3[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][6]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive2_4[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (520 * NB + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][7]);
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B1(n, 0, 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n] , send4, F1, NULL,NULL,NULL);
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send4[n][0], NDIM*(N3_GPU[n])*(N2_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (540 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
#endif
}

void Bp_send2(double(*restrict F2[NB])[NDIM], int n){
#if (MPI_enable)
	int j;
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive1_3[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][8]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive1_4[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][9]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive1_7[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][10]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive1_8[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (510 * NB + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][11]);
		}

		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B2(n, 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], send3, F2, NULL, NULL, NULL);
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send3[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (530 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive3_1[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][12]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			rc += MPI_Irecv(&receive3_2[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][13]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive3_5[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][14]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive3_6[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (530 * NB + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][15]);
		}

		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B2(n, 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], send1, F2, NULL, NULL, NULL);
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send1[n][0], NDIM*(N3_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (510 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
#endif
}

void Bp_send3(double(*restrict F3[NB])[NDIM], int n){
#if (MPI_enable)
	int z;
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive6_2[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][16]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive6_4[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][17]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive6_6[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][18]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive6_8[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (560 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][19]);
		}

		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B3(n, 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n], N2_GPU[n], send5, F3, NULL, NULL, NULL);
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send5[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (550 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive5_1[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][20]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			rc += MPI_Irecv(&receive5_3[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][21]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			rc += MPI_Irecv(&receive5_5[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][22]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive5_7[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (550 * NB + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][23]);
		}

		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B3(n, 0, N1_GPU[n], 0, N2_GPU[n], 0, D3, N1_GPU[n], N2_GPU[n], send6, F3, NULL, NULL, NULL);
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send6[n][0], NDIM*(N2_GPU[n])*(N1_GPU[n]), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (560 * NB + n)%MPI_TAG_MAX, mpi_cartcomm, &req[n]);
					MPI_Request_free(&req[n]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

void Bp_rec1(int n){
	int i;
	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][0], &Statbound[n][0]);
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive4_5[n][i] = send4[block[block[n][AMR_NBR2]][AMR_CHILD1]][i];
			}
		}

		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			MPI_Wait(&boundreqs[n][1], &Statbound[n][1]);
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_3 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive4_6[n][i] = send4[block[block[n][AMR_NBR2]][AMR_CHILD2]][i];
			}
		}

		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			MPI_Wait(&boundreqs[n][2], &Statbound[n][2]);
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive4_7[n][i] = send4[block[block[n][AMR_NBR2]][AMR_CHILD3]][i];
			}
		}

		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1){
			MPI_Wait(&boundreqs[n][3], &Statbound[n][3]);
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive4_8[n][i] = send4[block[block[n][AMR_NBR2]][AMR_CHILD4]][i];
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][4], &Statbound[n][4]);
		}
		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive2_1[n][i] = send2[block[block[n][AMR_NBR4]][AMR_CHILD5]][i];
			}
		}

		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			MPI_Wait(&boundreqs[n][5], &Statbound[n][5]);
		}
		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_3 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive2_2[n][i] = send2[block[block[n][AMR_NBR4]][AMR_CHILD6]][i];
			}
		}

		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			MPI_Wait(&boundreqs[n][6], &Statbound[n][6]);
		}
		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive2_3[n][i] = send2[block[block[n][AMR_NBR4]][AMR_CHILD7]][i];
			}
		}

		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1){
			MPI_Wait(&boundreqs[n][7], &Statbound[n][7]);
		}
		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1){
			for (i = 0; i < NDIM*(N3_GPU[n])*(N2_GPU[n]); i++){
				receive2_4[n][i] = send2[block[block[n][AMR_NBR4]][AMR_CHILD8]][i];
			}
		}
	}
}

void Bp_rec2(int n){
	int j;
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][8], &Statbound[n][8]);
		}
		else if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive1_3[n][j] = send1[block[block[n][AMR_NBR3]][AMR_CHILD1]][j];
			}
		}

		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			MPI_Wait(&boundreqs[n][9], &Statbound[n][9]);
		}
		else if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_3 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive1_4[n][j] = send1[block[block[n][AMR_NBR3]][AMR_CHILD2]][j];
			}
		}

		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			MPI_Wait(&boundreqs[n][10], &Statbound[n][10]);
		}
		else if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive1_7[n][j] = send1[block[block[n][AMR_NBR3]][AMR_CHILD5]][j];
			}
		}

		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			MPI_Wait(&boundreqs[n][11], &Statbound[n][11]);
		}
		else if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive1_8[n][j] = send1[block[block[n][AMR_NBR3]][AMR_CHILD6]][j];
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][12], &Statbound[n][12]);
		}
		else if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive3_1[n][j] = send3[block[block[n][AMR_NBR1]][AMR_CHILD3]][j];
			}
		}

		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1){
			MPI_Wait(&boundreqs[n][13], &Statbound[n][13]);
		}
		else if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_3 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive3_2[n][j] = send3[block[block[n][AMR_NBR1]][AMR_CHILD4]][j];
			}
		}

		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			MPI_Wait(&boundreqs[n][14], &Statbound[n][14]);
		}
		else if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive3_5[n][j] = send3[block[block[n][AMR_NBR1]][AMR_CHILD7]][j];
			}
		}

		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1){
			MPI_Wait(&boundreqs[n][15], &Statbound[n][15]);
		}
		else if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1){
			for (j = 0; j < NDIM*(N3_GPU[n])*(N1_GPU[n]); j++){
				receive3_6[n][j] = send3[block[block[n][AMR_NBR1]][AMR_CHILD8]][j];
			}
		}
	}
}

void Bp_rec3(int n){
	int z;
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][16], &Statbound[n][16]);
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive6_2[n][z] = send6[block[block[n][AMR_NBR5]][AMR_CHILD1]][z];
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			MPI_Wait(&boundreqs[n][17], &Statbound[n][17]);
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_2 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive6_4[n][z] = send6[block[block[n][AMR_NBR5]][AMR_CHILD3]][z];
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			MPI_Wait(&boundreqs[n][18], &Statbound[n][18]);
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive6_6[n][z] = send6[block[block[n][AMR_NBR5]][AMR_CHILD5]][z];
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			MPI_Wait(&boundreqs[n][19], &Statbound[n][19]);
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive6_8[n][z] = send6[block[block[n][AMR_NBR5]][AMR_CHILD7]][z];
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[n][20], &Statbound[n][20]);
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive5_1[n][z] = send5[block[block[n][AMR_NBR6]][AMR_CHILD2]][z];
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1){
			MPI_Wait(&boundreqs[n][21], &Statbound[n][21]);
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_2 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive5_3[n][z] = send5[block[block[n][AMR_NBR6]][AMR_CHILD4]][z];
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1){
			MPI_Wait(&boundreqs[n][22], &Statbound[n][22]);
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive5_5[n][z] = send5[block[block[n][AMR_NBR6]][AMR_CHILD6]][z];
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1){
			MPI_Wait(&boundreqs[n][23], &Statbound[n][23]);
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1){
			for (z = 0; z < NDIM*(N2_GPU[n])*(N1_GPU[n]); z++){
				receive5_7[n][z] = send5[block[block[n][AMR_NBR6]][AMR_CHILD8]][z];
			}
		}
	}
}