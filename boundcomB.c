#include "decs_MPI.h"

void pack_send_B1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					send[nl[n]][(i - i1)*zsize*jsize + (j - j1)*zsize + (z - z1)] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
		}
	}
}

void pack_send_B2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					send[nl[n]][(j - j1)*zsize*isize + (i - i1)*zsize + (z - z1)] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
		}
	}
}

void pack_send_B3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					send[nl[n]][(z - z1)*jsize*isize + (i - i1)*jsize + (j - j1)] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
		}
	}
}

void pack_send_B_average1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j += 1 + ref_2){
				for (z = z1; z < z2; z += (1 + ref_3)){
					send[nl[n]][(i - i1) *zsize*jsize + (j - j1) / (1 + ref_2)*zsize + (z - z1) / (1 + ref_3)]
						= 0.25*(F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][FACE1]
						+ F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][FACE1]);
				}
			}
		}
	}
}

void pack_send_B_average2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict F2[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i += 1 + ref_1){
				for (z = z1; z < z2; z += 1 + ref_3){
					send[nl[n]][(j - j1)*isize*zsize + (i - i1) / (1 + ref_1)*zsize + (z - z1) / (1 + ref_3)]
						= 0.25*(F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][FACE2] +
						F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][FACE2]);
				}
			}
		}
	}
}

void pack_send_B_average3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict F3[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3){
	if (gpu == 1){

	}
	else{
		int i, j, z, k;
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i += 1 + ref_1){
				for (j = j1; j < j2; j += 1 + ref_2){
					send[nl[n]][(z - z1)*isize*jsize + (i - i1) / (1 + ref_1)*jsize + (j - j1) / (1 + ref_2)]
						= 0.25*(F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + ref_1 + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]);
				}
			}
		}
	}
}


void unpack_receive_B1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor = 1.;
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					if (div == 1) factor = gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						= receive[nl[n_rec]][(i - i1)*zsize*jsize + (j - j1)*zsize + (z - z1)] / factor;
				}
			}
		}
	}
}

void unpack_receive_B2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent, int neg){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor = 1.;
		if (neg == 1)factor = -1.;
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					if (neg==0 && div == 1) factor = gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					else if (neg == 1 && div == 1) factor = -gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						= receive[nl[n_rec]][(j - j1)*zsize*isize + (i - i1)*zsize + (z - z1)] / factor;
				}
			}
		}
	}
}

void unpack_receive_B3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent){
	if (gpu == 1){

	}
	else{
		int i, j, z;
		double factor = 1.;
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					if (div == 1) factor = gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						= receive[nl[n_rec]][(z - z1)*isize*jsize + (i - i1)*jsize + (j - j1)] / factor;
				}
			}
		}
	}
}

/*Send boundaries between compute nodes through MPI*/
void B_send1(double(*restrict F1[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive4_fine[nl[n]][0], NDIM * (BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (44* NB_LOCAL + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][540]);
			}
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive4_5fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_1]][AMR_NODE], (44* NB_LOCAL + block[block[n][AMR_NBR2_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][545]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive4_6fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_2]][AMR_NODE], (44* NB_LOCAL + block[block[n][AMR_NBR2_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][546]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive4_7fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_3]][AMR_NODE], (44* NB_LOCAL + block[block[n][AMR_NBR2_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][547]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive4_8fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_4]][AMR_NODE], (44* NB_LOCAL + block[block[n][AMR_NBR2_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][548]);
		}
		if (block[n][AMR_NBR2P] >= 0){
			if (block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR2P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR2P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR2P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average1(n, BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_fine, F1, &(Bufferp[nl[n]]), &(Buffersend2fine[nl[n]]), &(boundevent[nl[n]][520]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR2P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send2_fine[nl[n]][0], NDIM*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR2P]][AMR_NODE], (42* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1){
			pack_send_B1(n, 0, 1, 0, BS_2, 0, BS_3,
				BS_2, BS_3, send4_fine, F1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[n]]), &(boundevent[nl[n]][540]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Isend(&send4_fine[nl[n]][0], NDIM*BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (44* NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive2_1fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_5]][AMR_NODE], (42* NB_LOCAL + block[block[n][AMR_NBR4_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][521]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive2_2fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_6]][AMR_NODE], (42* NB_LOCAL + block[block[n][AMR_NBR4_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][522]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive2_3fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_7]][AMR_NODE], (42* NB_LOCAL + block[block[n][AMR_NBR4_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][523]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE]&& ref_2 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive2_4fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_8]][AMR_NODE], (42* NB_LOCAL + block[block[n][AMR_NBR4_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][524]);
		}
		if (block[n][AMR_NBR4P] >= 0){
			if (block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR4P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR4P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR4P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average1(n, 0, 1, 0, BS_2, 0, BS_3,
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_fine, F1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[n]]), &(boundevent[nl[n]][540]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR4P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send4_fine[nl[n]][0], NDIM*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR4P]][AMR_NODE], (44* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void B_send2(double(*restrict F2[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] < NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2){
					pack_send_B2(n, 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
						BS_1, BS_3, send3_fine, F2, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[n]]), &(boundevent[nl[n]][530]));
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Isend(&send3_fine[nl[n]][0], NDIM*BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (43* NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						MPI_Request_free(&req[nl[n]]);
					}
				}
				else if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2){
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Irecv(&receive1_fine[nl[n]][0], NDIM * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (43* NB_LOCAL + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][510]);
					}
				}
			}
			else{
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Irecv(&receive1_fine[nl[n]][0], NDIM * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][510]);
				}
			}
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive1_3fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_1]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR3_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][513]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive1_4fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_2]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR3_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][514]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE]&& ref_1 == 1){
			rc += MPI_Irecv(&receive1_7fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_5]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR3_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][517]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive1_8fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_6]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR3_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][518]);
		}
		
		if (block[n][AMR_NBR3P] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			if (block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR3P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR3P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR3P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average2(n, 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_fine, F2, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[n]]), &(boundevent[nl[n]][530]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR3P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send3_fine[nl[n]][0], NDIM*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR3P]][AMR_NODE], (43* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){	
			if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] < NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2){
					pack_send_B2(n, 0, BS_1, 0, 1, 0, BS_3,
						BS_1, BS_3, send1_fine, F2, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[n]]), &(boundevent[nl[n]][510]));
					if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Isend(&send1_fine[nl[n]][0], NDIM*BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (41* NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						MPI_Request_free(&req[nl[n]]);
					}
				}
				else if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2){
					if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
						rc += MPI_Irecv(&receive3_fine[nl[n]][0], NDIM * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (41* NB_LOCAL + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][530]);
					}
				}
			}
			else{
				pack_send_B2(n, 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, send1_fine, F2, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[n]]), &(boundevent[nl[n]][510]));
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send1_fine[nl[n]][0], NDIM*BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (41* NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive3_1fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_3]][AMR_NODE], (43* NB_LOCAL + block[block[n][AMR_NBR1_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][531]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1 ){
			rc += MPI_Irecv(&receive3_2fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_4]][AMR_NODE], (43* NB_LOCAL + block[block[n][AMR_NBR1_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][532]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 ){
			rc += MPI_Irecv(&receive3_5fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_7]][AMR_NODE], (43* NB_LOCAL + block[block[n][AMR_NBR1_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][535]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive3_6fine[nl[n]][0], NDIM*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_8]][AMR_NODE], (43* NB_LOCAL + block[block[n][AMR_NBR1_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][536]);
		}
		if (block[n][AMR_NBR1P] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			if (block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR1P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR1P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR1P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average2(n, 0, BS_1, 0, 1, 0, BS_3,
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_fine, F2, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[n]]), &(boundevent[nl[n]][510]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR1P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send1_fine[nl[n]][0], NDIM*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR1P]][AMR_NODE], (41* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void B_send3(double(*restrict F3[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive6_fine[nl[n]][0], NDIM * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (46 * NB_LOCAL + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][560]);
			}
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive6_2fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_1]][AMR_NODE], (46* NB_LOCAL + block[block[n][AMR_NBR5_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][562]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive6_4fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_3]][AMR_NODE], (46* NB_LOCAL + block[block[n][AMR_NBR5_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][564]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive6_6fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_5]][AMR_NODE], (46* NB_LOCAL + block[block[n][AMR_NBR5_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][566]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			rc += MPI_Irecv(&receive6_8fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_7]][AMR_NODE], (46* NB_LOCAL + block[block[n][AMR_NBR5_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][568]);
		}
		if (block[n][AMR_NBR5P] >= 0){
			if (block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR5P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR5P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR5P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average3(n, 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_fine, F3, &(Bufferp[nl[n]]), &(Buffersend5fine[nl[n]]), &(boundevent[nl[n]][550]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR5P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send5_fine[nl[n]][0], NDIM*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR5P]][AMR_NODE], (45* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1){
			pack_send_B3(n, 0, BS_1, 0, BS_2, 0, D3, BS_1, BS_2, send6_fine, F3, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[n]]), &(boundevent[nl[n]][560]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Isend(&send6_fine[nl[n]][0], NDIM*BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (46* NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive5_1fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_2]][AMR_NODE], (45* NB_LOCAL + block[block[n][AMR_NBR6_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][551]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive5_3fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_4]][AMR_NODE], (45* NB_LOCAL + block[block[n][AMR_NBR6_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][553]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive5_5fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_6]][AMR_NODE], (45* NB_LOCAL + block[block[n][AMR_NBR6_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][555]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			rc += MPI_Irecv(&receive5_7fine[nl[n]][0], NDIM*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_8]][AMR_NODE], (45* NB_LOCAL + block[block[n][AMR_NBR6_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][557]);
		}
		if (block[n][AMR_NBR6P] >= 0){
			if (block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR6P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR6P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR6P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_B_average3(n, 0, BS_1, 0, BS_2, 0, D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_fine, F3, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[n]]), &(boundevent[nl[n]][560]), ref_1, ref_2, ref_3);
				if (block[block[n][AMR_NBR6P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send6_fine[nl[n]][0], NDIM*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR6P]][AMR_NODE], (46* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}


/*Receive boundaries for compute nodes through MPI*/
void B_rec1(double(*restrict F1[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][521], &Statbound[nl[n]][521]);
				unpack_receive_B1(n, n, 0, 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_1fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec2_1fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B1(n, block[n][AMR_NBR4_5], 0, 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend2fine[nl[block[n][AMR_NBR4_5]]]), &(boundevent[nl[block[n][AMR_NBR4_5]]][520]));
			}
			set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][522], &Statbound[nl[n]][522]);
					unpack_receive_B1(n, n, 0, 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_2fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec2_2fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR4_6], 0, 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend2fine[nl[block[n][AMR_NBR4_6]]]), &(boundevent[nl[block[n][AMR_NBR4_6]]][520]));
				}
			}
			set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][523], &Statbound[nl[n]][523]);
					unpack_receive_B1(n, n, 0, 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_3fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec2_3fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR4_7], 0, 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend2fine[nl[block[n][AMR_NBR4_7]]]), &(boundevent[nl[block[n][AMR_NBR4_7]]][520]));
				}
			}
			set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
			if (ref_2 && ref_3 == 1){
				if (block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][524], &Statbound[nl[n]][524]);
					unpack_receive_B1(n, n, 0, 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_4fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec2_4fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR4_8], 0, 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend2fine[nl[block[n][AMR_NBR4_8]]]), &(boundevent[nl[block[n][AMR_NBR4_8]]][520]));
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][540], &Statbound[nl[n]][540]);
				unpack_receive_B1(n, n, BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, receive4_fine, F1, 0, &(Bufferp[nl[n]]), &(Bufferrec4fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B1(n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, send4_fine, F1, 0, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[block[n][AMR_NBR2]]]), &(boundevent[nl[block[n][AMR_NBR2]]][540]));
			}		
		}	
		else if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][545], &Statbound[nl[n]][545]);
				unpack_receive_B1(n, n, BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_5fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec4_5fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B1(n, block[n][AMR_NBR2_1], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[block[n][AMR_NBR2_1]]]), &(boundevent[nl[block[n][AMR_NBR2_1]]][540]));
			}
			set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][546], &Statbound[nl[n]][546]);
					unpack_receive_B1(n, n, BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_6fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec4_6fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR2_2], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[block[n][AMR_NBR2_2]]]), &(boundevent[nl[block[n][AMR_NBR2_2]]][540]));
				}
			}
			set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][547], &Statbound[nl[n]][547]);
					unpack_receive_B1(n, n, BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_7fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec4_7fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR2_3], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[block[n][AMR_NBR2_3]]]), &(boundevent[nl[block[n][AMR_NBR2_3]]][540]));
				}
			}
			set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
			if (ref_2==1 && ref_3 == 1){
				if (block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][548], &Statbound[nl[n]][548]);
					unpack_receive_B1(n, n, BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_8fine, F1, 1, &(Bufferp[nl[n]]), &(Bufferrec4_8fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B1(n, block[n][AMR_NBR2_4], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_fine, F1, 1, &(Bufferp[nl[n]]), &(Buffersend4fine[nl[block[n][AMR_NBR2_4]]]), &(boundevent[nl[block[n][AMR_NBR2_4]]][540]));
				}
			}
		}
	}
#endif
}

void B_rec2(double(*restrict F2[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && block[n][AMR_COORD3] >=NB_3*pow(1+REF_3,block[n][AMR_LEVEL3])/2){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][530], &Statbound[nl[n]][530]);
					unpack_receive_B2(n, n, 0, BS_1, 0, 1, 0, BS_3,
						BS_1, BS_3, receive3_fine, F2, 0, &(Bufferp[nl[n]]), &(Bufferrec3fine[nl[n]]), NULL, 1);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
						BS_1, BS_3, send1_fine, F2, 0, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR1]]]), &(boundevent[nl[block[n][AMR_NBR1]]][510]), 1);
				}
			}
			else{
			
			}
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][531], &Statbound[nl[n]][531]);
				unpack_receive_B2(n, n, 0, BS_1 / (1 + ref_1), 0, 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_1fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec3_1fine[nl[n]]), NULL, 0);
			}
			else{
				unpack_receive_B2(n, block[n][AMR_NBR1_3], 0, BS_1 / (1 + ref_1), 0, 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[block[n][AMR_NBR1_3]]]), &(boundevent[nl[block[n][AMR_NBR1_3]]][530]), 0);
			}
			set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][532], &Statbound[nl[n]][532]);
					unpack_receive_B2(n, n, 0, BS_1 / (1 + ref_1), 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_2fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec3_2fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR1_4], 0, BS_1 / (1 + ref_1), 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[block[n][AMR_NBR1_4]]]), &(boundevent[nl[block[n][AMR_NBR1_4]]][530]), 0);
				}
			}
			set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][535], &Statbound[nl[n]][535]);
					unpack_receive_B2(n, n, BS_1 / (1 + ref_1), BS_1, 0, 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_5fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec3_5fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR1_7], BS_1 / (1 + ref_1), BS_1, 0, 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[block[n][AMR_NBR1_7]]]), &(boundevent[nl[block[n][AMR_NBR1_7]]][530]), 0);
				}
			}
			set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
			if (ref_1==1 && ref_3 == 1){
				if (block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][536], &Statbound[nl[n]][536]);
					unpack_receive_B2(n, n, BS_1 / (1 + ref_1), BS_1, 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_6fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec3_6fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR1_8], BS_1 / (1 + ref_1), BS_1, 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[block[n][AMR_NBR1_8]]]), &(boundevent[nl[block[n][AMR_NBR1_8]]][530]), 0);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
				if (block[n][AMR_COORD3] >= NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2){
					if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
						MPI_Wait(&boundreqs[nl[n]][510], &Statbound[nl[n]][510]);
						unpack_receive_B2(n, n, 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
							BS_1, BS_3, receive1_fine, F2, 0, &(Bufferp[nl[n]]), &(Bufferrec1fine[nl[n]]), NULL, 1);
					}
					else{
						unpack_receive_B2(n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
							BS_1, BS_3, send3_fine, F2, 0, &(Bufferp[nl[n]]), &(Buffersend3fine[nl[block[n][AMR_NBR3]]]), &(boundevent[nl[block[n][AMR_NBR3]]][530]), 1);
					}
				}
			}
			else{
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][510], &Statbound[nl[n]][510]);
					unpack_receive_B2(n, n, 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
						BS_1, BS_3, receive1_fine, F2, 0, &(Bufferp[nl[n]]), &(Bufferrec1fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
						BS_1, BS_3, send1_fine, F2, 0, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR3]]]), &(boundevent[nl[block[n][AMR_NBR3]]][510]), 0);
				}
			}
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][513], &Statbound[nl[n]][513]);
				unpack_receive_B2(n, n, 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_3fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec1_3fine[nl[n]]), NULL, 0);
			}
			else{
				unpack_receive_B2(n, block[n][AMR_NBR3_1], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR3_1]]]), &(boundevent[nl[block[n][AMR_NBR3_1]]][510]), 0);
			}
			set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][514], &Statbound[nl[n]][514]);
					unpack_receive_B2(n, n, 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_4fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec1_4fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR3_2], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR3_2]]]), &(boundevent[nl[block[n][AMR_NBR3_2]]][510]), 0);
				}
			}
			set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][517], &Statbound[nl[n]][517]);
					unpack_receive_B2(n, n, BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_7fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec1_7fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR3_5], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR3_5]]]), &(boundevent[nl[block[n][AMR_NBR3_5]]][510]), 0);
				}
			}
			set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
			if (ref_1==1 && ref_3 == 1){
				if (block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][518], &Statbound[nl[n]][518]);
					unpack_receive_B2(n, n, BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_8fine, F2, 1, &(Bufferp[nl[n]]), &(Bufferrec1_8fine[nl[n]]), NULL, 0);
				}
				else{
					unpack_receive_B2(n, block[n][AMR_NBR3_6], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_fine, F2, 1, &(Bufferp[nl[n]]), &(Buffersend1fine[nl[block[n][AMR_NBR3_6]]]), &(boundevent[nl[block[n][AMR_NBR3_6]]][510]), 0);
				}
			}
		}
	}
#endif
}
void B_rec3(double(*restrict F3[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][551], &Statbound[nl[n]][551]);
				unpack_receive_B3(n, n, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_1fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec5_1fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B3(n, block[n][AMR_NBR6_2], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend5fine[nl[block[n][AMR_NBR6_2]]]), &(boundevent[nl[block[n][AMR_NBR6_2]]][550]));
			}
			set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][553], &Statbound[nl[n]][553]);
					unpack_receive_B3(n, n, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_3fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec5_3fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR6_4], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend5fine[nl[block[n][AMR_NBR6_4]]]), &(boundevent[nl[block[n][AMR_NBR6_4]]][550]));
				}
			}
			set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][555], &Statbound[nl[n]][555]);
					unpack_receive_B3(n, n, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_5fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec5_5fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR6_6], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend5fine[nl[block[n][AMR_NBR6_6]]]), &(boundevent[nl[block[n][AMR_NBR6_6]]][550]));
				}
			}
			set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
			if (ref_1==1 && ref_2 == 1){
				if (block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][557], &Statbound[nl[n]][557]);
					unpack_receive_B3(n, n, BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_7fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec5_7fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR6_8], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend5fine[nl[block[n][AMR_NBR6_8]]]), &(boundevent[nl[block[n][AMR_NBR6_8]]][550]));
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][560], &Statbound[nl[n]][560]);
				unpack_receive_B3(n, n, 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, receive6_fine, F3, 0, &(Bufferp[nl[n]]), &(Bufferrec6fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B3(n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, send6_fine, F3, 0, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[block[n][AMR_NBR5]]]), &(boundevent[nl[block[n][AMR_NBR5]]][560]));
			}
		}
		else if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]){
				MPI_Wait(&boundreqs[nl[n]][562], &Statbound[nl[n]][562]);
				unpack_receive_B3(n, n, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_2fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec6_2fine[nl[n]]), NULL);
			}
			else{
				unpack_receive_B3(n, block[n][AMR_NBR5_1], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[block[n][AMR_NBR5_1]]]), &(boundevent[nl[block[n][AMR_NBR5_1]]][560]));
			}
			set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][564], &Statbound[nl[n]][564]);
					unpack_receive_B3(n, n, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_4fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec6_4fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR5_3], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[block[n][AMR_NBR5_3]]]), &(boundevent[nl[block[n][AMR_NBR5_3]]][560]));
				}
			}
			set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][566], &Statbound[nl[n]][566]);
					unpack_receive_B3(n, n, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_6fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec6_6fine[nl[n]]),NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR5_5], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[block[n][AMR_NBR5_5]]]), &(boundevent[nl[block[n][AMR_NBR5_5]]][560]));
				}
			}
			set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
			if (ref_1==1 && ref_2 == 1){
				if (block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE]){
					MPI_Wait(&boundreqs[nl[n]][568], &Statbound[nl[n]][568]);
					unpack_receive_B3(n, n, BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_8fine, F3, 1, &(Bufferp[nl[n]]), &(Bufferrec6_8fine[nl[n]]), NULL);
				}
				else{
					unpack_receive_B3(n, block[n][AMR_NBR5_7], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_fine, F3, 1, &(Bufferp[nl[n]]), &(Buffersend6fine[nl[block[n][AMR_NBR5_7]]]), &(boundevent[nl[block[n][AMR_NBR5_7]]][560]));
				}
			}
		}
	}
		//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Send boundaries between compute nodes through MPI*/
void Bp_send1(double(*restrict F1[NB_LOCAL])[NDIM], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int i;
	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive4_5[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR2_1]][AMR_NODE], (34* NB_LOCAL + block[block[n][AMR_NBR2_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][100]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive4_6[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR2_2]][AMR_NODE], (34* NB_LOCAL + block[block[n][AMR_NBR2_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][101]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive4_7[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR2_3]][AMR_NODE], (34* NB_LOCAL + block[block[n][AMR_NBR2_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][102]);
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive4_8[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR2_4]][AMR_NODE], (34* NB_LOCAL + block[block[n][AMR_NBR2_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][103]);
		}
		if (block[n][AMR_NBR2P] >= 0){
			if (block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B1(n, BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, send2, F1, NULL, NULL, NULL);
				if (block[block[n][AMR_NBR2P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send2[nl[n]][0], NDIM*(BS_3)*(BS_2) , MPI_DOUBLE, block[block[n][AMR_NBR2P]][AMR_NODE], (32* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive2_1[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR4_5]][AMR_NODE], (32* NB_LOCAL + block[block[n][AMR_NBR4_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][104]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive2_2[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR4_6]][AMR_NODE], (32* NB_LOCAL + block[block[n][AMR_NBR4_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][105]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive2_3[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR4_7]][AMR_NODE], (32* NB_LOCAL + block[block[n][AMR_NBR4_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][106]);
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive2_4[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR4_8]][AMR_NODE], (32* NB_LOCAL + block[block[n][AMR_NBR4_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][107]);
		}
		if (block[n][AMR_NBR4P] >= 0){
			if (block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B1(n, 0, 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3 , send4, F1, NULL,NULL,NULL);
				if (block[block[n][AMR_NBR4P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send4[nl[n]][0], NDIM*(BS_3)*(BS_2), MPI_DOUBLE, block[block[n][AMR_NBR4P]][AMR_NODE], (34* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void Bp_send2(double(*restrict F2[NB_LOCAL])[NDIM], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int j;
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive1_3[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR3_1]][AMR_NODE], (31* NB_LOCAL + block[block[n][AMR_NBR3_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][108]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive1_4[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR3_2]][AMR_NODE], (31* NB_LOCAL + block[block[n][AMR_NBR3_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][109]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive1_7[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR3_5]][AMR_NODE], (31* NB_LOCAL + block[block[n][AMR_NBR3_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][110]);
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive1_8[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR3_6]][AMR_NODE], (31* NB_LOCAL + block[block[n][AMR_NBR3_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][111]);
		}

		if (block[n][AMR_NBR3P] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){
			if (block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B2(n, 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1, BS_3, send3, F2, NULL, NULL, NULL);
				if (block[block[n][AMR_NBR3P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send3[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR3P]][AMR_NODE], (33* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive3_1[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR1_3]][AMR_NODE], (33* NB_LOCAL + block[block[n][AMR_NBR1_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][112]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			rc += MPI_Irecv(&receive3_2[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR1_4]][AMR_NODE], (33* NB_LOCAL + block[block[n][AMR_NBR1_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][113]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive3_5[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR1_7]][AMR_NODE], (33* NB_LOCAL + block[block[n][AMR_NBR1_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][114]);
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			rc += MPI_Irecv(&receive3_6[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR1_8]][AMR_NODE], (33* NB_LOCAL + block[block[n][AMR_NBR1_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][115]);
		}

		if (block[n][AMR_NBR1P] >= 0 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){
			if (block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B2(n, 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, send1, F2, NULL, NULL, NULL);
				if (block[block[n][AMR_NBR1P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send1[nl[n]][0], NDIM*(BS_3)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR1P]][AMR_NODE], (31* NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void Bp_send3(double(*restrict F3[NB_LOCAL])[NDIM], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int z;
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive6_2[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR5_1]][AMR_NODE], (36* NB_LOCAL + block[block[n][AMR_NBR5_1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][116]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive6_4[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR5_3]][AMR_NODE], (36* NB_LOCAL + block[block[n][AMR_NBR5_3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][117]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive6_6[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR5_5]][AMR_NODE], (36* NB_LOCAL + block[block[n][AMR_NBR5_5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][118]);
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			rc += MPI_Irecv(&receive6_8[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR5_7]][AMR_NODE], (36* NB_LOCAL + block[block[n][AMR_NBR5_7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][119]);
		}

		if (block[n][AMR_NBR5P] >= 0){
			if (block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B3(n, 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, send5, F3, NULL, NULL, NULL);
				if (block[block[n][AMR_NBR5P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send5[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR5P]][AMR_NODE], (35 * NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]){
			rc += MPI_Irecv(&receive5_1[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR6_2]][AMR_NODE], (35 * NB_LOCAL + block[block[n][AMR_NBR6_2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][120]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			rc += MPI_Irecv(&receive5_3[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR6_4]][AMR_NODE], (35 * NB_LOCAL + block[block[n][AMR_NBR6_4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][121]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			rc += MPI_Irecv(&receive5_5[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR6_6]][AMR_NODE], (35 * NB_LOCAL + block[block[n][AMR_NBR6_6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][122]);
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			rc += MPI_Irecv(&receive5_7[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR6_8]][AMR_NODE], (35 * NB_LOCAL + block[block[n][AMR_NBR6_8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][123]);
		}

		if (block[n][AMR_NBR6P] >= 0){
			if (block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_B3(n, 0, BS_1, 0, BS_2, 0, D3, BS_1, BS_2, send6, F3, NULL, NULL, NULL);
				if (block[block[n][AMR_NBR6P]][AMR_NODE] != block[n][AMR_NODE]){
					rc += MPI_Isend(&send6[nl[n]][0], NDIM*(BS_2)*(BS_1), MPI_DOUBLE, block[block[n][AMR_NBR6P]][AMR_NODE], (36 * NB_LOCAL + block[n][AMR_NUMBER])%MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

void Bp_rec1(int n){
	int ref_1, ref_2, ref_3;

	int i;
	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][100], &Statbound[nl[n]][100]);
		}
		else if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive4_5[nl[n]][i] = send4[nl[block[n][AMR_NBR2_1]]][i];
			}
		}

		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][101], &Statbound[nl[n]][101]);
		}
		else if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && ref_3 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive4_6[nl[n]][i] = send4[nl[block[n][AMR_NBR2_2]]][i];
			}
		}

		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][102], &Statbound[nl[n]][102]);
		}
		else if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && ref_2 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive4_7[nl[n]][i] = send4[nl[block[n][AMR_NBR2_3]]][i];
			}
		}

		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1 && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][103], &Statbound[nl[n]][103]);
		}
		else if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && ref_2 == 1 && ref_3 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive4_8[nl[n]][i] = send4[nl[block[n][AMR_NBR2_4]]][i];
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][104], &Statbound[nl[n]][104]);
		}
		else if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive2_1[nl[n]][i] = send2[nl[block[n][AMR_NBR4_5]]][i];
			}
		}

		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][105], &Statbound[nl[n]][105]);
		}
		else if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && ref_3 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive2_2[nl[n]][i] = send2[nl[block[n][AMR_NBR4_6]]][i];
			}
		}

		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][106], &Statbound[nl[n]][106]);
		}
		else if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && ref_2 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive2_3[nl[n]][i] = send2[nl[block[n][AMR_NBR4_7]]][i];
			}
		}

		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1 && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][107], &Statbound[nl[n]][107]);
		}
		else if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && ref_2 == 1 && ref_3 == 1){
			for (i = 0; i < NDIM*(BS_3)*(BS_2); i++){
				receive2_4[nl[n]][i] = send2[nl[block[n][AMR_NBR4_8]]][i];
			}
		}
	}
}

void Bp_rec2(int n){
	int ref_1, ref_2, ref_3;
	int j;
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][108], &Statbound[nl[n]][108]);
		}
		else if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive1_3[nl[n]][j] = send1[nl[block[n][AMR_NBR3_1]]][j];
			}
		}

		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][109], &Statbound[nl[n]][109]);
		}
		else if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && ref_3 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive1_4[nl[n]][j] = send1[nl[block[n][AMR_NBR3_2]]][j];
			}
		}

		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			MPI_Wait(&boundreqs[nl[n]][110], &Statbound[nl[n]][110]);
		}
		else if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && ref_1 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive1_7[nl[n]][j] = send1[nl[block[n][AMR_NBR3_5]]][j];
			}
		}

		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][111], &Statbound[nl[n]][111]);
		}
		else if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && ref_1 == 1 && ref_3 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive1_8[nl[n]][j] = send1[nl[block[n][AMR_NBR3_6]]][j];
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][112], &Statbound[nl[n]][112]);
		}
		else if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive3_1[nl[n]][j] = send3[nl[block[n][AMR_NBR1_3]]][j];
			}
		}

		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][113], &Statbound[nl[n]][113]);
		}
		else if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && ref_3 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive3_2[nl[n]][j] = send3[nl[block[n][AMR_NBR1_4]]][j];
			}
		}

		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			MPI_Wait(&boundreqs[nl[n]][114], &Statbound[nl[n]][114]);
		}
		else if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && ref_1 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive3_5[nl[n]][j] = send3[nl[block[n][AMR_NBR1_7]]][j];
			}
		}

		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_3 == 1){
			MPI_Wait(&boundreqs[nl[n]][115], &Statbound[nl[n]][115]);
		}
		else if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && ref_1 == 1 && ref_3 == 1){
			for (j = 0; j < NDIM*(BS_3)*(BS_1); j++){
				receive3_6[nl[n]][j] = send3[nl[block[n][AMR_NBR1_8]]][j];
			}
		}
	}
}

void Bp_rec3(int n){
	int ref_1, ref_2, ref_3;
	int z;
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][116], &Statbound[nl[n]][116]);
		}
		else if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive6_2[nl[n]][z] = send6[nl[block[n][AMR_NBR5_1]]][z];
			}
		}

		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][117], &Statbound[nl[n]][117]);
		}
		else if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && ref_2 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive6_4[nl[n]][z] = send6[nl[block[n][AMR_NBR5_3]]][z];
			}
		}

		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			MPI_Wait(&boundreqs[nl[n]][118], &Statbound[nl[n]][118]);
		}
		else if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && ref_1 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive6_6[nl[n]][z] = send6[nl[block[n][AMR_NBR5_5]]][z];
			}
		}

		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][119], &Statbound[nl[n]][119]);
		}
		else if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && ref_1 == 1 && ref_2 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive6_8[nl[n]][z] = send6[nl[block[n][AMR_NBR5_7]]][z];
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]){
			MPI_Wait(&boundreqs[nl[n]][120], &Statbound[nl[n]][120]);
		}
		else if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive5_1[nl[n]][z] = send5[nl[block[n][AMR_NBR6_2]]][z];
			}
		}

		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][121], &Statbound[nl[n]][121]);
		}
		else if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && ref_2 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive5_3[nl[n]][z] = send5[nl[block[n][AMR_NBR6_4]]][z];
			}
		}

		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1){
			MPI_Wait(&boundreqs[nl[n]][122], &Statbound[nl[n]][122]);
		}
		else if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && ref_1 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive5_5[nl[n]][z] = send5[nl[block[n][AMR_NBR6_6]]][z];
			}
		}

		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1 && ref_2 == 1){
			MPI_Wait(&boundreqs[nl[n]][123], &Statbound[nl[n]][123]);
		}
		else if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && ref_1 == 1 && ref_2 == 1){
			for (z = 0; z < NDIM*(BS_2)*(BS_1); z++){
				receive5_7[nl[n]][z] = send5[nl[block[n][AMR_NBR6_8]]][z];
			}
		}
	}
}