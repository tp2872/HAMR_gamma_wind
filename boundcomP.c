#include "decs.h"

void pack_send1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1)*(z2 - z1);
		status = packsend1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);
		cudaEventRecord(boundevent1[0],commandQueueGPU[n]);
		if (status != 0)printf("error pack_send1 %d", status);
	}
	else{
		int i, j, z, k;
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + k] = prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (0 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[n][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (1 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[n][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (2 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					#endif
				}
			}
		}
	}
}

void pack_send2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(z2 - z1);
		status = packsend2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);
		cudaEventRecord(boundevent1[0], commandQueueGPU[n]);
		if (status != 0)printf("error pack_send2 %d",status);
	}
	else{
		int i, j, z, k;
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + k] = prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (0 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[n][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (1 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[n][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (2 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];

					#endif
				}
			}
		}
	}
}

void pack_send3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		status = packsend3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);
		cudaEventRecord(boundevent1[0],commandQueueGPU[n]);
		if (status != 0)printf("error pack_send3 %d", status);
	}
	else{
		int i, j, z, k;
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + k] = prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (0 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[n][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (1 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[n][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (2 + NPR)] = 
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					#endif
				}
			}
		}
	}
}

void pack_send_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3);
		status = packsendaverage1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);
		cudaEventRecord(boundevent1[0],commandQueueGPU[n]);
		if (status != 0)printf("error pack_send_average1 %d", status);
	}
	else{
		int i, j, z, k;
		for (i = i1; i < i2; i += 1 + REF_1){
			for (j = j1; j < j2; j += 1 + REF_2){
				for (z = z1; z < z2; z += (1 + REF_3)){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + k]
							= 0.125*(prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (0 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE1]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE1] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][FACE1]);

					send[n][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (1 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE2]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE2]);

					send[n][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (2 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]);
					#endif
				}
			}
		}
	}
}

void pack_send_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3);
		status = packsendaverage2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);
		cudaEventRecord(boundevent1[0],commandQueueGPU[n]);
		if (status != 0)printf("error pack_send_average2 %d", status);
	}
	else{
		int i, j, z, k;
		for (j = j1; j < j2; j += 1 + REF_2){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (z = z1; z < z2; z += 1 + REF_3){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + k]
							= 0.125*(prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (0 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE1]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n])][FACE1] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n]+REF_3)][FACE1]);

					send[n][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (1 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n] + REF_3)][FACE2]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][FACE2] +
						ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] , z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] , z + N3_GPU_offset[n] + REF_3)][FACE2]);

					send[n][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (2 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n])][FACE3]);
					#endif
				}
			}
		}
	}
}

void pack_send_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferps, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1) / (1 + REF_1)*(j2 - j1) / (1 + REF_2)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + REF_1)*(j2 - j1) / (1 + REF_2);
		status = packsendaverage3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[n], work_size);		
		cudaEventRecord(boundevent1[0], commandQueueGPU[n]);
		if (status != 0)printf("error pack_send_average3 %d", status);
	}
	else{
		int i, j, z, k;
		for (z = z1; z < z2; z += 1 + REF_3){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (j = j1; j < j2; j += 1 + REF_2){
					for (k = 0; k < NPR; k++){
						send[n][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + k]
							= 0.125*(prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
					#if(STAGGERED)
					send[n][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (0 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE1]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n])][FACE1]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n]+REF_2, z + N3_GPU_offset[n]+REF_3)][FACE1]);
					send[n][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (1 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE2]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[n][index2(n, i + N1_GPU_offset[n]+REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n]+REF_3)][FACE2]);
					send[n][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (2 + NPR)] =
						0.25*(ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] )][3] * gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] )][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] )][3] * gdet[n][index2(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[n][index2(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]);
					#endif
				}
			}
		}
	}
}

void unpack_receive1(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *tempBufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int mpi){
	double Bf1, Bf2, Bf3;
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL) cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (j2 - j1)*(z2 - z1);
		status = unpackreceive1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, jsize, zsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferboundreceive[0], tempBufferboundreceive[0],
			update_staggered, Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0) printf("unpack_receive1 error! %d \n",status);
	}
	else{
		int i, j, z, k;
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
						ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
					}
					#if(STAGGERED)
					if (update_staggered == 1 && (z<0 || z >= BS_3 || j<0 || j >= BS_2)){
						ps[n][index(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)] /
							gdet[n][index2(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						psh[n][index(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)] /
							gdet[n][index2(n, i + N1_GPU_offset[n] , j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					}
					else psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)] /
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					#endif
				}
			}
		}
	}
}

void unpack_receive2(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int isize, int zsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *tempBufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int reverse){
	double factor=1.;
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL) cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (i2 - i1)*(z2 - z1);
		status = unpackreceive2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, isize, zsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferboundreceive[0], tempBufferboundreceive[0],
			reverse, update_staggered, Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0) printf("unpack_receive2 error! \n");
	}
	else{
		int i, j, z, k;
		if (reverse == 0){
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						for (k = 0; k < NPR; k++){
							p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
							ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
						}
						#if(STAGGERED)
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)]/
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)]/
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						if (update_staggered == 1 && (i<0 || i >= BS_1 || z<0 || z >= BS_3)){
							ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)] /
								gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][FACE2];
							psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)] /
								gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] , z + N3_GPU_offset[n])][FACE2];
						}
						else psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)] /
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)]/
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
						psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)]/
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
						#endif
					}
				}
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				for (k = 0; k < NPR; k++){
					if (k == U3 || k == U2 || k == B2 || k == B3) factor = -1.;
					else factor = 1.;
					p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = factor*receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
					ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = factor*receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + k];
				}
				/*double ucon[NDIM], utcon[NDIM];
				struct of_geom geom;
				
				if (j<0)get_geometry(n, i + N1_GPU_offset[n], -j-1 + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				else get_geometry(n, i + N1_GPU_offset[n], 2*BS_2-j-1 + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				get_geometry(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				ucon_calc(p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])], &geom, ucon);
				ucon[3] *= -1.;
				ucon[2] *= -1.;
				
				//get_geometry(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);

				ucon_to_utcon(ucon, &geom, utcon);
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U2] = utcon[2];
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U3] = utcon[3];
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U2] = utcon[2];
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U3] = utcon[3];

				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B2] *= -1.;
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B3] *= -1.;
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B2] *= -1.;
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B3] *= -1.;
				*/
				#if(STAGGERED)
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)];
				if (nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || nstep == -1){
					//ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = -receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)];
				}
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = -receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (0 + NPR)];
				//psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = -receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (1 + NPR)];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = -receive[n_rec2][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3)) + (2 + NPR)];
				#endif
			}
		}
	}
}

void unpack_receive3(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int isize, int jsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *tempBufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		clSetKernelArg(kernel_unpackreceive3[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive3[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive3[n], 2, sizeof(cl_int), &i_offset);
		clSetKernelArg(kernel_unpackreceive3[n], 3, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive3[n], 4, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive3[n], 5, sizeof(cl_int), &j_offset);
		clSetKernelArg(kernel_unpackreceive3[n], 6, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive3[n], 7, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive3[n], 8, sizeof(cl_int), &z_offset);
		clSetKernelArg(kernel_unpackreceive3[n], 9, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_unpackreceive3[n], 10, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_unpackreceive3[n], 11, sizeof(cl_mem), (void *)&(Bufferp_1[n]));
		clSetKernelArg(kernel_unpackreceive3[n], 12, sizeof(cl_mem), (void *)&(Bufferph_1[n]));
		clSetKernelArg(kernel_unpackreceive3[n], 13, sizeof(cl_mem), (void *)&(Bufferps_1[n]));
		clSetKernelArg(kernel_unpackreceive3[n], 14, sizeof(cl_mem), (void *)&(Bufferpsh_1[n]));
		clSetKernelArg(kernel_unpackreceive3[n], 15, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive3[n], 16, sizeof(cl_mem), (void *)&(tempBufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive3[n], 17, sizeof(cl_int), &update_staggered);
		clSetKernelArg(kernel_unpackreceive3[n], 18, sizeof(cl_mem), (void *)&Imagegdet[n]);
		clSetKernelArg(kernel_unpackreceive3[n], 19, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive3[n], 20, sizeof(cl_double), &dt);
		clSetKernelArg(kernel_unpackreceive3[n], 21, sizeof(cl_int), &block[n][AMR_TIMELEVEL]);
		clSetKernelArg(kernel_unpackreceive3[n], 22, sizeof(cl_int), &block[n_rec][AMR_TIMELEVEL]);

		int nr_workgroups_bound = ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL) cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (i2 - i1)*(j2 - j1);
		status = unpackreceive3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, isize, jsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferboundreceive[0], tempBufferboundreceive[0],
			update_staggered, Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0) printf("unpack_receive3 error! \n");
	}
	else{
		int i, j, z, k;
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (k = 0; k < NPR; k++){
						p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + k];
						ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + k];
					}
					#if(STAGGERED)
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (0 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (0 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (1 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (1 + NPR)]/
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					if (update_staggered == 1 && (i<0 || i >= BS_1  || j<0 || j >= BS_2)){
						ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (2 + NPR)] /
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
						psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (2 + NPR)] /
							gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] )][FACE3];
					}
					else psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*(z - z1 + z_offset * N3G / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * N1G / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * N2G / (1 + REF_2)) + (2 + NPR)] /
						gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					#endif
				}
			}
		}
	}
}

void unpack_receive_coarse1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *temp1Bufferboundreceive, cl_mem *temp2Bufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL)cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (j2 - j1)*(z2 - z1);
		status = unpackreceivecoarse1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferp[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0)printf("error unpack_receive_coarse1: %d \n", status);
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1,dq2,dq3, avg;
		for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
			//Use slope limited interpolation in direction fluxes, copy  boundary cells in other directions
			if (i1 < 0 && REF_1 == 1) ii = REF_1;
			else if (REF_1 == 1)ii = 0;
			else ii = i - i1;
			ij = (j - j1 - (j - j1) % (1 + REF_2)) / (1 + REF_2) + REF_2;
			iz = (z - z1 - (z - z1) % (1 + REF_3)) / (1 + REF_3) + REF_3;

			is = ((i == i1) ? (-1) : (1));
			js = (((j - j1) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = (((z - z1) % (1 + REF_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR; k++){
				avg = 0.5*(prim[n][index(n, N1_GPU_offset[n] + (1 - ii)*(N1_GPU[n] - 1), j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] + prim[n][index(n, ii + N1_GPU_offset[n] + (1 - ii)*(N1_GPU[n] - 2), j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				if (ii == 0){
					dq1 = slope_lim(avg, receive[n_rec2][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+k], receive[n_rec2][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + k]);
				}
				else{
					dq1 = slope_lim(receive[n_rec2][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+k], receive[n_rec2][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + k], avg);
				}
				dq2 = slope_lim(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij-REF_2)*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij+REF_2)*zsize + (NPR + 3)*iz + k]);
				dq3 = slope_lim(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz-REF_3) + k], receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz-REF_3) + k]);
				dq2 = dq3= 0.;
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}
			#if(STAGGERED)
			if (js == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (1 + NPR)] + receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + (1 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (1 + NPR)] + receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + (1 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (1 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (1 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			if (zs == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)]+receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*(iz+REF_3) + (2 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
			#endif
		}
	}
}

void unpack_receive_coarse2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *temp1Bufferboundreceive, cl_mem *temp2Bufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL) cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (i2 - i1)*(z2 - z1);
		status = unpackreceivecoarse2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferp[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0)printf("error unpack_receive_coarse2");
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1, dq2, dq3, avg;
		for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
			//now use zero order interpolation, must be done better in the future
			if (j1 < 0 && REF_2 == 1) ij = REF_2;
			else if (REF_2 == 1) ij = 0;
			else ij = j - j1;
			ii = (i - i1 - (i - i1) % (1 + REF_1)) / (1 + REF_1) + REF_1;
			iz = (z - z1 - (z - z1) % (1 + REF_3)) / (1 + REF_3) + REF_3;
			
			is = (((i - i1) % (1 + REF_1) == 0) ? (-1) : (1));
			js = ((j == j1) ? (-1) : (1));
			zs = (((z - z1) % (1 + REF_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR; k++){	
				avg = 0.5*(prim[n][index(n, i+ N1_GPU_offset[n], N2_GPU_offset[n] + (1 - ij)*(N2_GPU[n] - 1), z + N3_GPU_offset[n])][k] + prim[n][index(n,i + N1_GPU_offset[n], ij + N2_GPU_offset[n] + (1 - ij)*(N2_GPU[n] - 2), z + N3_GPU_offset[n])][k]);
				if (ij == 0){
					dq2 = slope_lim(avg, receive[n_rec2][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+k], receive[n_rec2][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + k]);
				}
				else{
					dq2 = slope_lim(receive[n_rec2][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+k], receive[n_rec2][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + k], avg);
				}
				dq1 = slope_lim(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii-REF_1)*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii+REF_1)*zsize + (NPR + 3)*iz + k]);
				dq3 = slope_lim(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz-REF_3) + k], receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k], receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz+REF_3) + k]);
				dq1 = dq3 = 0.;
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}
			#if(STAGGERED)
			if (is == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + receive[n_rec2][(NPR + 3)*(ij)*zsize*isize + (NPR + 3)*(ii+REF_1)*zsize + (NPR + 3)*iz + (0 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + receive[n_rec2][(NPR + 3)*(ij)*zsize*isize + (NPR + 3)*(ii+REF_1)*zsize + (NPR + 3)*iz + (0 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			if (zs == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*(iz+REF_3) + (2 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[n_rec2][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
			#endif
		}
	}
}

void unpack_receive_coarse3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *temp1Bufferboundreceive, cl_mem *temp2Bufferboundreceive, cl_event *boundevent1, cl_event *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = ceil((double)((i2 - i1)*(j2 - j1))/((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1){
			if (boundevent1 != NULL)cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[0], 0);
		}
		int work_size = (i2 - i1)*(j2 - j1);
		status = unpackreceivecoarse3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[n] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp_1[n], Bufferph_1[n], Bufferps_1[n], Bufferpsh_1[n], Bufferp[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[n], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		if (status != 0)printf("error unpack_receive_coarse3");
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1,dq2,dq3, avg;
		for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
			//now use zero order interpolation, must be done better in the future
			if (z1 < 0 && REF_3 == 1) iz = REF_3;
			else if (REF_3 == 1) iz = 0;
			else iz = z - z1;
			ij = (j - j1 - (j - j1) % (1 + REF_2)) / (1 + REF_2) + REF_2;
			ii = (i - i1 - (i - i1) % (1 + REF_1)) / (1 + REF_1) + REF_1;
			
			is = (((i - i1) % (1 + REF_1) == 0) ? (-1) : (1));
			js = (((j - j1) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = ((z == z1) ? (-1) : (1));

			for (k = 0; k < NPR; k++){
				avg = 0.5*(prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], N3_GPU_offset[n] + (1 - iz)*(N3_GPU[n] - 1))][k] + prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], iz + N3_GPU_offset[n] + (1 - iz)*(N3_GPU[n] - 2))][k]);
				if (iz == 0){
					dq3 = slope_lim(avg, receive[n_rec2][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+k], receive[n_rec2][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + k]);
				}
				else{
					dq3 = slope_lim(receive[n_rec2][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+k], receive[n_rec2][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + k], avg);
				}
				dq1 = slope_lim(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii-REF_1)*jsize + (NPR + 3)*ij + k], receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k], receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii+REF_1)*jsize + (NPR + 3)*ij + k]);
				dq2 = slope_lim(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij-REF_2) + k], receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k], receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij+REF_2) + k]);
				dq1 = dq2 = 0.;
				ph[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}
			#if(STAGGERED)
			if (is == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii+REF_1)*jsize + (NPR + 3)*ij + (0 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii + REF_1)*jsize + (NPR + 3)*ij + (0 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			if (js == 1){
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*(ij+REF_2) + (1 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*(ij + REF_2) + (1 + NPR)])/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			else{
				ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[n_rec2][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)]/
					gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			#endif
		}
	}
}

/*Send boundaries between compute nodes through MPI*/
void bound_send1(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem Bufferp[NB], cl_mem Bufferps[NB], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X1

	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send1(n, block[n][AMR_NBR2], N1_GPU[n] - N1G, N1_GPU[n], -N2G, N2_GPU[n] + N2G, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), send2, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend2[n]),
				&(boundevent1[n][20]), &(boundevent2[n][20]));
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (40 * NB + block[n][AMR_NBR2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][40]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][20],0);
					rc += MPI_Isend(&Buffersend2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (20 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				}
				else{
					rc += MPI_Irecv(&receive4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (40 * NB + block[n][AMR_NBR2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][40]);
					rc += MPI_Isend(&send2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (20 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				}
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send2_1 to finer grid
			pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n] - N1G, N1_GPU[n], -N2G, N2_GPU[n] / (1 + REF_2) + N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_1, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend2_1[n]),
				&(boundevent1[n][21]), &(boundevent2[n][21]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec4_5[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][45]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][21],0);
					rc += MPI_Isend(&Buffersend2_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (21 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[1]);
				}
				else{
					rc += MPI_Irecv(&receive4_5[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][45]);
					rc += MPI_Isend(&send2_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (21 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[1]);
				}
				MPI_Request_free(&req[1]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send2_2 to finer grid
			pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n] - N1G, N1_GPU[n], -N2G, N2_GPU[n] / (1 + REF_2) + N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_2, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend2_2[n]),
				&(boundevent1[n][22]), &(boundevent2[n][22]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec4_6[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][46]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][22],0);
					rc += MPI_Isend(&Buffersend2_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (22 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[2]);
				}
				else{
					rc += MPI_Irecv(&receive4_6[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][46]);
					rc += MPI_Isend(&send2_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (22 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[2]);
				}
				MPI_Request_free(&req[2]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send2_3 to finer grid
			pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n] - N1G, N1_GPU[n], N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_3, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend2_3[n]),
				&(boundevent1[n][23]), &(boundevent2[n][23]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec4_7[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][47]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][23],0);
					rc += MPI_Isend(&Buffersend2_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (23 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[3]);
				}
				else{
					rc += MPI_Irecv(&receive4_7[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][47]);
					rc += MPI_Isend(&send2_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (23 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[3]);
				}
				MPI_Request_free(&req[3]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send2_4 to finer grid
			pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n] - N1G, N1_GPU[n], N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_4, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend2_4[n]),
				&(boundevent1[n][24]), &(boundevent2[n][24]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec4_8[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][48]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][24],0);
					rc += MPI_Isend(&Buffersend2_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (24 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[4]);
				}
				else{
					rc += MPI_Irecv(&receive4_8[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (40 * NB + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][48]);
					rc += MPI_Isend(&send2_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (24 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[4]);
				}
				MPI_Request_free(&req[4]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send to coarser grid
				pack_send_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n] - (1 + REF_1) * N1G, N1_GPU[n], -N2G, N2_GPU[n] + N2G, -N3G, N3_GPU[n] + N3G,
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send2, prim,ps, &(Bufferp[n]), &(Bufferps[n]),
					&(Buffersend2[n]), &(boundevent1[n][20]), &(boundevent2[n][20]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
							rc += MPI_Irecv(&Bufferrec4_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (45 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][45]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
							rc += MPI_Irecv(&Bufferrec4_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (46 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][46]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec4_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (47 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][47]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
							rc += MPI_Irecv(&Bufferrec4_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (48 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][48]);
						}
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][20],0);
						rc += MPI_Isend(&Buffersend2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (20 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[5]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
							rc += MPI_Irecv(&receive4_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (45 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][45]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
							rc += MPI_Irecv(&receive4_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (46 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][46]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
							rc += MPI_Irecv(&receive4_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (47 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][47]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
							rc += MPI_Irecv(&receive4_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (48 * NB + block[block[n][AMR_NBR2]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][48]);
						}
						rc += MPI_Isend(&send2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (20 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[5]);
					}
					MPI_Request_free(&req[5]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send1(n, block[n][AMR_NBR4], 0, N1G, -N2G, N2_GPU[n] + N2G, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), send4, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4[n]),
				&(boundevent1[n][40]), &(boundevent2[n][40]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (20 * NB + block[n][AMR_NBR4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][20]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][40],0);
					rc += MPI_Isend(&Buffersend4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (40 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[59]);
				}
				else{
					rc += MPI_Irecv(&receive2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (20 * NB + block[n][AMR_NBR4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][20]);
					rc += MPI_Isend(&send4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (40 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[59]);
				}
				MPI_Request_free(&req[59]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send4_5 to finer grid
			pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_5, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4_5[n]),
				&(boundevent1[n][45]), &(boundevent2[n][45]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec2_1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][21]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][45],0);
					rc += MPI_Isend(&Buffersend4_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (45 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[6]);
				}
				else{
					rc += MPI_Irecv(&receive2_1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][21]);
					rc += MPI_Isend(&send4_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (45 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[6]);
				}
				MPI_Request_free(&req[6]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send4_6 to finer grid
			pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_6, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4_6[n]),
				&(boundevent1[n][46]), &(boundevent2[n][46]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec2_2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][22]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][46],0);
					rc += MPI_Isend(&Buffersend4_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (46 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[7]);
				}
				else{
					rc += MPI_Irecv(&receive2_2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][22]);
					rc += MPI_Isend(&send4_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (46 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[7]);
				}
				MPI_Request_free(&req[7]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send4_7 to finer grid
			pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_7, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4_7[n]),
				&(boundevent1[n][47]), &(boundevent2[n][47]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec2_3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][23]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][47],0);
					rc += MPI_Isend(&Buffersend4_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (47 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[8]);
				}
				else{
					rc += MPI_Irecv(&receive2_3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][23]);
					rc += MPI_Isend(&send4_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (47 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[8]);
				}
				MPI_Request_free(&req[8]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send4_8 to finer grid
			pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
				(N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_8, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4_8[n]),
				&(boundevent1[n][48]), &(boundevent2[n][48]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec2_4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][24]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][48],0);
					rc += MPI_Isend(&Buffersend4_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (48 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[9]);
				}
				else{
					rc += MPI_Irecv(&receive2_4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (20 * NB + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][24]);
					rc += MPI_Isend(&send4_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (48 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[9]);
				}
				MPI_Request_free(&req[9]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send to coarser grid
				pack_send_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, (1 + REF_1)*N1G, -N2G, (N2_GPU[n] + N2G), -N3G, (N3_GPU[n] + N3G),
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send4, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend4[n]),
					&(boundevent1[n][40]), &(boundevent2[n][40]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							rc += MPI_Irecv(&Bufferrec2_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (21 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][21]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
							rc += MPI_Irecv(&Bufferrec2_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (22 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][22]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec2_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (23 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][23]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
							rc += MPI_Irecv(&Bufferrec2_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (24 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][24]);
						}
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][40],0);
						rc += MPI_Isend(&Buffersend4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (40 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[10]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							rc += MPI_Irecv(&receive2_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (21 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][21]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
							rc += MPI_Irecv(&receive2_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (22 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][22]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							rc += MPI_Irecv(&receive2_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (23 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][23]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
							rc += MPI_Irecv(&receive2_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (24 * NB + block[block[n][AMR_NBR4]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][24]);
						}
						rc += MPI_Isend(&send4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (40 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[10]);
					}
					MPI_Request_free(&req[10]);
				}
			}
		}
	}
#endif
}

void bound_send2(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem Bufferp[NB], cl_mem Bufferps[NB], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send2(n, block[n][AMR_NBR3], -N1G, N1_GPU[n] + N1G, N2_GPU[n] - N2G, N2_GPU[n], -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send3, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3[n]),
				&(boundevent1[n][30]), &(boundevent2[n][30]));
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){

				if (gpu == 1){
					if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
						rc += MPI_Irecv(&Bufferrec1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][10]);
					}
					else{
						rc += MPI_Irecv(&Bufferrec1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (10 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][10]);
					}
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][30],0);
					rc += MPI_Isend(&Buffersend3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[60]);
				}
				else{
					if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
						rc += MPI_Irecv(&receive1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][10]);
					}
					else{
						rc += MPI_Irecv(&receive1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (10 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][10]);
					}
					rc += MPI_Isend(&send3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[60]);
				}
				MPI_Request_free(&req[60]);
			}
		}
		if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send3_1 to finer grid
				pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, N2_GPU[n] - N2G, N2_GPU[n], -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_1, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3_1[n]),
					&(boundevent1[n][31]), &(boundevent2[n][31]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec1_3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][13]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][31],0);
						rc += MPI_Isend(&Buffersend3_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (31 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[11]);
					}
					else{
						rc += MPI_Irecv(&receive1_3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][13]);
						rc += MPI_Isend(&send3_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (31 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[11]);
					}
					MPI_Request_free(&req[11]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send3_2 to finer grid
				pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, N2_GPU[n] - N2G, N2_GPU[n], N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_2, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3_2[n]),
					&(boundevent1[n][32]), &(boundevent2[n][32]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec1_4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][14]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][32],0);
						rc += MPI_Isend(&Buffersend3_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (32 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[12]);
					}
					else{
						rc += MPI_Irecv(&receive1_4[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][14]);
						rc += MPI_Isend(&send3_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (32 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[12]);
					}
					MPI_Request_free(&req[12]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send3_5 to finer grid
				pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, N2_GPU[n] - N2G, N2_GPU[n], -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_5, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3_5[n]),
					&(boundevent1[n][35]), &(boundevent2[n][35]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec1_7[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][17]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][35],0);
						rc += MPI_Isend(&Buffersend3_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (35 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[13]);
					}
					else{
						rc += MPI_Irecv(&receive1_7[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][17]);
						rc += MPI_Isend(&send3_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (35 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[13]);
					}
					MPI_Request_free(&req[13]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send3_6 to finer grid
				pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, N2_GPU[n] - N2G, N2_GPU[n], N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_6, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3_6[n]),
					&(boundevent1[n][36]), &(boundevent2[n][36]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec1_8[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][18]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][36],0);
						rc += MPI_Isend(&Buffersend3_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (36 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[14]);
					}
					else{
						rc += MPI_Irecv(&receive1_8[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (10 * NB + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][18]);
						rc += MPI_Isend(&send3_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (36 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[14]);
					}
					MPI_Request_free(&req[14]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
					//send to coarser grid
					pack_send_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, (N1_GPU[n] + N1G), (N2_GPU[n] - (1 + REF_2) * N2G), N2_GPU[n], -N3G, (N3_GPU[n] + N3G),
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send3, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend3[n]),
						&(boundevent1[n][30]), &(boundevent2[n][30]));
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
							if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
								rc += MPI_Irecv(&Bufferrec1_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (13 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][13]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
								rc += MPI_Irecv(&Bufferrec1_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (14 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][14]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
								rc += MPI_Irecv(&Bufferrec1_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (17 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][17]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
								rc += MPI_Irecv(&Bufferrec1_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (18 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][18]);
							}
							cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][30],0);
							rc += MPI_Isend(&Buffersend3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (30 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[15]);
						}
						else{
							if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
								rc += MPI_Irecv(&receive1_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (13 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][13]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
								rc += MPI_Irecv(&receive1_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (14 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][14]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
								rc += MPI_Irecv(&receive1_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (17 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][17]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
								rc += MPI_Irecv(&receive1_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (18 * NB + block[block[n][AMR_NBR3]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][18]);
							}
							rc += MPI_Isend(&send3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (30 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[15]);
						}
						MPI_Request_free(&req[15]);
					}
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send2(n, block[n][AMR_NBR1], -N1G, N1_GPU[n] + N1G, 0, N2G, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send1, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1[n]),
				&(boundevent1[n][10]), &(boundevent2[n][10]));
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
						rc += MPI_Irecv(&Bufferrec3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][30]);
					}
					else{
						rc += MPI_Irecv(&Bufferrec3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (30 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][30]);
					}
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][10],0);
					rc += MPI_Isend(&Buffersend1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[16]);
				}
				else{
					if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
						rc += MPI_Irecv(&receive3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][30]);
					}
					else{
						rc += MPI_Irecv(&receive3[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (30 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][30]);
					}
					rc += MPI_Isend(&send1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G)*(N1_GPU[n] + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[16]);
				}
				MPI_Request_free(&req[16]);
			}
		}
		if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2){
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send1_3 to finer grid
				pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, 0, N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_3, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1_3[n]),
					&(boundevent1[n][13]), &(boundevent2[n][13]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec3_1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][31]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][13],0);
						rc += MPI_Isend(&Buffersend1_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (13 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[17]);
					}
					else{
						rc += MPI_Irecv(&receive3_1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][31]);
						rc += MPI_Isend(&send1_3[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (13 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[17]);
					}
					MPI_Request_free(&req[17]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send1_4 to finer grid
				pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, 0, N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_4, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1_4[n]),
					&(boundevent1[n][14]), &(boundevent2[n][14]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec3_2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][32]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][14],0);
						rc += MPI_Isend(&Buffersend1_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (14 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[18]);
					}
					else{
						rc += MPI_Irecv(&receive3_2[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][32]);
						rc += MPI_Isend(&send1_4[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (14 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[18]);
					}
					MPI_Request_free(&req[18]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send1_7 to finer grid
				pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, 0, N2G, -N3G, N3_GPU[n] / (1 + REF_3) + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_7, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1_7[n]),
					&(boundevent1[n][17]), &(boundevent2[n][17]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec3_5[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][35]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][17],0);
						rc += MPI_Isend(&Buffersend1_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (17 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[19]);
					}
					else{
						rc += MPI_Irecv(&receive3_5[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][35]);
						rc += MPI_Isend(&send1_7[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (17 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[19]);
					}
					MPI_Request_free(&req[19]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send1_8 to finer grid
				pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, 0, N2G, N3_GPU[n] / (1 + REF_3) - N3G, N3_GPU[n] + N3G,
					(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_8, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1_8[n]),
					&(boundevent1[n][18]), &(boundevent2[n][18]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						rc += MPI_Irecv(&Bufferrec3_6[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][36]);
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][18],0);
						rc += MPI_Isend(&Buffersend1_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (18 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[20]);
					}
					else{
						rc += MPI_Irecv(&receive3_6[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (30 * NB + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][36]);
						rc += MPI_Isend(&send1_8[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (18 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[20]);
					}
					MPI_Request_free(&req[20]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
					//send to coarser grid
					pack_send_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, 0, (1 + REF_2) * N2G, -N3G, N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send1, prim, ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend1[n]),
						&(boundevent1[n][10]), &(boundevent2[n][10]));
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
							if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
								rc += MPI_Irecv(&Bufferrec3_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (31 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][31]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
								rc += MPI_Irecv(&Bufferrec3_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (32 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][32]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
								rc += MPI_Irecv(&Bufferrec3_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (35 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][35]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
								rc += MPI_Irecv(&Bufferrec3_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (36 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][36]);
							}
							cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][10]);
							rc += MPI_Isend(&Buffersend1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (10 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[21]);
						}
						else{
							if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
								rc += MPI_Irecv(&receive3_1[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (31 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][31]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
								rc += MPI_Irecv(&receive3_2[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (32 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][32]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
								rc += MPI_Irecv(&receive3_5[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (35 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][35]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
								rc += MPI_Irecv(&receive3_6[n][0], (NPR + 3)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (36 * NB + block[block[n][AMR_NBR1]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][36]);
							}
							rc += MPI_Isend(&send1[n][0], (NPR + 3)*(N3_GPU[n] + 2 * N3G) / (1 + REF_3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (10 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[21]);
						}
						MPI_Request_free(&req[21]);
					}
				}
			}
		}
	}
#endif
}

void bound_send3(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], cl_mem Bufferp[NB], cl_mem Bufferps[NB], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send3(n, block[n][AMR_NBR5], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G, N3_GPU[n] - N3G, N3_GPU[n], (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), send5, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5[n]),
				&(boundevent1[n][50]), &(boundevent2[n][50]));
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (60 * NB + block[n][AMR_NBR5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][60]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][50],0);
					rc += MPI_Isend(&Buffersend5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (50 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[22]);
				}
				else{
					rc += MPI_Irecv(&receive6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (60 * NB + block[n][AMR_NBR5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][60]);
					rc += MPI_Isend(&send5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (50 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[22]);
				}
				MPI_Request_free(&req[22]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send5_1 to finer grid
			pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, N3_GPU[n] - N3G, N3_GPU[n],
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_1, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5_1[n]),
				&(boundevent1[n][51]), &(boundevent2[n][51]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec6_2[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][62]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][51],0);
					rc += MPI_Isend(&Buffersend5_1[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (51 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[23]);
				}
				else{
					rc += MPI_Irecv(&receive6_2[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][62]);
					rc += MPI_Isend(&send5_1[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (51 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[23]);
				}
				MPI_Request_free(&req[23]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send5_3 to finer grid
			pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, N3_GPU[n] - N3G, N3_GPU[n],
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_3, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5_3[n]),
				&(boundevent1[n][53]), &(boundevent2[n][53]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec6_4[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][64]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][53],0);
					rc += MPI_Isend(&Buffersend5_3[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (53 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[24]);
				}
				else{
					rc += MPI_Irecv(&receive6_4[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][64]);
					rc += MPI_Isend(&send5_3[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (53 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[24]);
				}
				MPI_Request_free(&req[24]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send5_5 to finer grid
			pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, N3_GPU[n] - N3G, N3_GPU[n],
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_5, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5_5[n]),
				&(boundevent1[n][55]), &(boundevent2[n][55]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec6_6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][66]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][55],0);
					rc += MPI_Isend(&Buffersend5_5[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (55 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[25]);
				}
				else{
					rc += MPI_Irecv(&receive6_6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][66]);
					rc += MPI_Isend(&send5_5[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (55 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[25]);
				}
				MPI_Request_free(&req[25]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send5_7 to finer grid
			pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, N3_GPU[n] - N3G, N3_GPU[n],
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_7, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5_7[n]),
				&(boundevent1[n][57]), &(boundevent2[n][57]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive6_8[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][68]);
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec6_8[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][68]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][57],0);
					rc += MPI_Isend(&Buffersend5_7[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (57 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[26]);
				}
				else{
					rc += MPI_Irecv(&receive6_8[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (60 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][68]);
					rc += MPI_Isend(&send5_7[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (57 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[26]);
				}
				MPI_Request_free(&req[26]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send to coarser grid
				pack_send_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, (N1_GPU[n] + N1G), -N2G, (N2_GPU[n] + N2G), N3_GPU[n] - (1 + REF_3) * N3G, N3_GPU[n],
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send5, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend5[n]),
					&(boundevent1[n][50]), &(boundevent2[n][50]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){

					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
							rc += MPI_Irecv(&Bufferrec6_2[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (62 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][62]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec6_4[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (64 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][64]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
							rc += MPI_Irecv(&Bufferrec6_6[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (66 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][66]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec6_8[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (68 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][68]);
						}
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][50]);
						rc += MPI_Isend(&Buffersend5[n][0], (NPR + 3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (50 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[27]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
							rc += MPI_Irecv(&receive6_2[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (62 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][62]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
							rc += MPI_Irecv(&receive6_4[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (64 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][64]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
							rc += MPI_Irecv(&receive6_6[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (66 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][66]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
							rc += MPI_Irecv(&receive6_8[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (68 * NB + block[block[n][AMR_NBR5]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][68]);
						}
						rc += MPI_Isend(&send5[n][0], (NPR + 3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (50 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[27]);
					}
					MPI_Request_free(&req[27]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			pack_send3(n, block[n][AMR_NBR6], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G, 0, N3G, (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), send6, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6[n]),
				&(boundevent1[n][60]), &(boundevent2[n][60]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				rc += MPI_Irecv(&receive5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (50 * NB + block[n][AMR_NBR6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][50]);
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (50 * NB + block[n][AMR_NBR6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][50]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][60],0);
					rc += MPI_Isend(&Buffersend6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (60 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[28]);
				}
				else{
					rc += MPI_Irecv(&receive5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (50 * NB + block[n][AMR_NBR6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][50]);
					rc += MPI_Isend(&send6[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (60 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[28]);
				}
				MPI_Request_free(&req[28]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send6_2 to finer grid
			pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, 0, N3G,
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_2, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6_2[n]),
				&(boundevent1[n][62]), &(boundevent2[n][62]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec5_1[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][51]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][62],0);
					rc += MPI_Isend(&Buffersend6_2[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (62 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[29]);
				}
				else{
					rc += MPI_Irecv(&receive5_1[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][51]);
					rc += MPI_Isend(&send6_2[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (62 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[29]);
				}
				MPI_Request_free(&req[29]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send6_4 to finer grid
			pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], -N1G, N1_GPU[n] / (1 + REF_1) + N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, 0, N3G,
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_4, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6_4[n]),
				&(boundevent1[n][64]), &(boundevent2[n][64]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec5_3[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][53]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][64],0);
					rc += MPI_Isend(&Buffersend6_4[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (64 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[30]);
				}
				else{
					rc += MPI_Irecv(&receive5_3[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][53]);
					rc += MPI_Isend(&send6_4[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (64 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[30]);
				}
				MPI_Request_free(&req[30]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send6_6 to finer grid
			pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] / (1 + REF_2) + N2G, 0, N3G,
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_6, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6_6[n]),
				&(boundevent1[n][66]), &(boundevent2[n][66]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec5_5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][55]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][66],0);
					rc += MPI_Isend(&Buffersend6_6[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (66 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[31]);
				}
				else{
					rc += MPI_Irecv(&receive5_5[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][55]);
					rc += MPI_Isend(&send6_6[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (66 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[31]);
				}
				MPI_Request_free(&req[31]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1)){
			//send6_8 to finer grid
			pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1) - N1G, N1_GPU[n] + N1G, N2_GPU[n] / (1 + REF_2) - N2G, N2_GPU[n] + N2G, 0, N3G,
				(N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_8, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6_8[n]),
				&(boundevent1[n][68]), &(boundevent2[n][68]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					rc += MPI_Irecv(&Bufferrec5_7[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][57]);
					cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][68],0);
					rc += MPI_Isend(&Buffersend6_8[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (68 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[32]);
				}
				else{
					rc += MPI_Irecv(&receive5_7[n][0], (NPR + 3)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (50 * NB + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][57]);
					rc += MPI_Isend(&send6_8[n][0], (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (68 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[32]);
				}
				MPI_Request_free(&req[32]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
				//send to coarser grid
				pack_send_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, (N1_GPU[n] + N1G), -N2G, (N2_GPU[n] + N2G), 0, (1 + REF_3) * N3G,
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send6, prim,ps, &(Bufferp[n]), &(Bufferps[n]), &(Buffersend6[n]),
					&(boundevent1[n][60]), &(boundevent2[n][60]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							rc += MPI_Irecv(&Bufferrec5_1[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (51 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][51]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec5_3[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (53 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][53]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
							rc += MPI_Irecv(&Bufferrec5_5[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (55 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][55]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
							rc += MPI_Irecv(&Bufferrec5_7[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (57 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][57]);
						}
						cudaStreamWaitEvent(commandQueueGPU[n], boundevent1[n][60],0);
						rc += MPI_Isend(&send6[n][0], (NPR + 3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (60 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[33]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							rc += MPI_Irecv(&receive5_1[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (51 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][51]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							rc += MPI_Irecv(&receive5_3[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (53 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][53]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
							rc += MPI_Irecv(&receive5_5[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (55 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][55]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
							rc += MPI_Irecv(&receive5_7[n][0], (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (57 * NB + block[block[n][AMR_NBR6]][AMR_PARENT]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][57]);
						}
						rc += MPI_Isend(&send6[n][0], (NPR + 3)*(N2_GPU[n] + 2 * N2G) / (1 + REF_2)*(N1_GPU[n] + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (60 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[33]);
					}
					MPI_Request_free(&req[33]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void bound_rec1(double(*restrict prim[NB])[NPR], cl_mem Bufferp[NB], int bound_force, int n){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][20], &Statbound[n][20]);
				}
				unpack_receive1(n, block[n][AMR_NBR4], 0, -N1G, 0, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), receive2, tempreceive2, prim,
					&(Bufferp[n]), &(Bufferrec2[n]), &(tempBufferrec2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[n][AMR_NBR4], 0, -N1G, 0, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), send2, tempreceive2, prim,
					&(Bufferp[n]), &(Buffersend2[block[n][AMR_NBR4]]), &(tempBufferrec2[n]), &(boundevent1[block[n][AMR_NBR4]][20]), &(boundevent2[block[n][AMR_NBR4]][20]), block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
			}
		}

		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][21], &Statbound[n][21]);
				}
				unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, -N1G, 0, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive2_1, tempreceive2_1, prim,
					&(Bufferp[n]), &(Bufferrec2_1[n]), &(tempBufferrec2_1[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, -N1G, 0, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send2, tempreceive2_1, prim,
					&(Bufferp[n]), &(Buffersend2[block[block[n][AMR_NBR4]][AMR_CHILD5]]), &(tempBufferrec2_1[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_CHILD5]][20]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_CHILD5]][20]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][22], &Statbound[n][22]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, -N1G, 0, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive2_2, tempreceive2_2, prim,
						&(Bufferp[n]), &(Bufferrec2_2[n]), &(tempBufferrec2_2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, -N1G, 0, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send2, tempreceive2_2, prim,
						&(Bufferp[n]), &(Buffersend2[block[block[n][AMR_NBR4]][AMR_CHILD6]]), &(tempBufferrec2_2[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_CHILD6]][20]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_CHILD6]][20]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][23], &Statbound[n][23]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, -N1G, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + REF_2*D2, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive2_3, tempreceive2_3, prim,
						&(Bufferp[n]), &(Bufferrec2_3[n]), &(tempBufferrec2_3[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, -N1G, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + REF_2*D2, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send2, tempreceive2_3, prim,
						&(Bufferp[n]), &(Buffersend2[block[block[n][AMR_NBR4]][AMR_CHILD7]]), &(tempBufferrec2_3[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_CHILD7]][20]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_CHILD7]][20]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][24], &Statbound[n][24]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, -N1G, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + REF_2*D2, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive2_4, tempreceive2_4, prim,
						&(Bufferp[n]), &(Bufferrec2_4[n]), &(tempBufferrec2_4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, -N1G, 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + REF_2*D2, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send2, tempreceive2_4, prim,
						&(Bufferp[n]), &(Buffersend2[block[block[n][AMR_NBR4]][AMR_CHILD8]]), &(tempBufferrec2_4[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_CHILD8]][20]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_CHILD8]][20]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][21], &Statbound[n][21]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive2_1, send4_5, tempreceive2_1, prim,
						&(Bufferp[n]), &(Bufferrec2_1[n]), &(Buffersend4_5[n]), &(tempBufferrec2_1[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_1, receive2_1, tempreceive2_1, prim,
						&(Bufferp[n]), &(Buffersend2_1[block[block[n][AMR_NBR4]][AMR_PARENT]]), &(Bufferrec2_1[n]), &(tempBufferrec2_1[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_PARENT]][21]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_PARENT]][21]), block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][22], &Statbound[n][22]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive2_2, send4_6, tempreceive2_2, prim,
						&(Bufferp[n]), &(Bufferrec2_2[n]), &(Buffersend4_6[n]), &(tempBufferrec2_2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_2, receive2_2, tempreceive2_2, prim,
						&(Bufferp[n]), &(Buffersend2_2[block[block[n][AMR_NBR4]][AMR_PARENT]]), &(Bufferrec2_2[n]), &(tempBufferrec2_2[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_PARENT]][22]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_PARENT]][22]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][23], &Statbound[n][23]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive2_3, send4_7, tempreceive2_3, prim,
						&(Bufferp[n]), &(Bufferrec2_3[n]), &(Buffersend4_7[n]), &(tempBufferrec2_3[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_3, receive2_3, tempreceive2_3, prim,
						&(Bufferp[n]), &(Buffersend2_3[block[block[n][AMR_NBR4]][AMR_PARENT]]), &(Bufferrec2_3[n]), &(tempBufferrec2_3[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_PARENT]][23]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_PARENT]][23]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][24], &Statbound[n][24]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive2_4, send4_8, tempreceive2_4, prim,
						&(Bufferp[n]), &(Bufferrec2_4[n]), &(Buffersend4_8[n]), &(tempBufferrec2_4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, (N2_GPU[n] / (1 + REF_2) + 2 * N2G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send2_4, receive2_4, tempreceive2_4, prim,
						&(Bufferp[n]), &(Buffersend2_4[block[block[n][AMR_NBR4]][AMR_PARENT]]), &(Bufferrec2_4[n]), &(tempBufferrec2_4[n]), &(boundevent1[block[block[n][AMR_NBR4]][AMR_PARENT]][24]), &(boundevent2[block[block[n][AMR_NBR4]][AMR_PARENT]][24]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else printf("Error in indexing!\n");
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][40], &Statbound[n][40]);
				}
				unpack_receive1(n, block[n][AMR_NBR2], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), receive4, tempreceive4, prim,
					&(Bufferp[n]), &(Bufferrec4[n]), &(tempBufferrec4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[n][AMR_NBR2], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N2_GPU[n] + 2 * N2G), (N3_GPU[n] + 2 * N3G), send4, tempreceive4, prim,
					&(Bufferp[n]), &(Buffersend4[block[n][AMR_NBR2]]), &(tempBufferrec4[n]), &(boundevent1[block[n][AMR_NBR2]][40]), &(boundevent2[block[n][AMR_NBR2]][40]), block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][45], &Statbound[n][45]);
				}
				unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive4_5, tempreceive4_5, prim,
					&(Bufferp[n]), &(Bufferrec4_5[n]), &(tempBufferrec4_5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
					(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send4, tempreceive4_5, prim,
					&(Bufferp[n]), &(Buffersend4[block[block[n][AMR_NBR2]][AMR_CHILD1]]), &(tempBufferrec4_5[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_CHILD1]][40]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_CHILD1]][40]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][46], &Statbound[n][46]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive4_6, tempreceive4_6, prim,
						&(Bufferp[n]), &(Bufferrec4_6[n]), &(tempBufferrec4_6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], 0, N1_GPU[n], N1_GPU[n] + N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + REF_3*D3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send4, tempreceive4_6, prim,
						&(Bufferp[n]), &(Buffersend4[block[block[n][AMR_NBR2]][AMR_CHILD2]]), &(tempBufferrec4_6[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_CHILD2]][40]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_CHILD2]][40]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][47], &Statbound[n][47]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], 0, N1_GPU[n], N1_GPU[n] + N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive4_7, tempreceive4_7, prim,
						&(Bufferp[n]), &(Bufferrec4_7[n]), &(tempBufferrec4_7[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], 0, N1_GPU[n], N1_GPU[n] + N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send4, tempreceive4_7, prim,
						&(Bufferp[n]), &(Buffersend4[block[block[n][AMR_NBR2]][AMR_CHILD3]]), &(tempBufferrec4_7[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_CHILD3]][40]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_CHILD3]][40]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][48], &Statbound[n][48]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], 0, N1_GPU[n], N1_GPU[n] + N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive4_8, tempreceive4_8, prim,
						&(Bufferp[n]), &(Bufferrec4_8[n]), &(tempBufferrec4_8[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], 0, N1_GPU[n], N1_GPU[n] + N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
						(N2_GPU[n] + 2 * N2G) / (1 + REF_2), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send4, tempreceive4_8, prim,
						&(Bufferp[n]), &(Buffersend4[block[block[n][AMR_NBR2]][AMR_CHILD4]]), &(tempBufferrec4_8[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_CHILD4]][40]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_CHILD4]][40]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
						MPI_Wait(&boundreqs[n][45], &Statbound[n][45]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive4_5, send2_1, tempreceive4_5, prim,
						&(Bufferp[n]), &(Bufferrec4_5[n]), &(Buffersend2_1[n]), &(tempBufferrec4_5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_5, receive4_5, tempreceive4_5, prim,
						&(Bufferp[n]), &(Buffersend4_5[block[block[n][AMR_NBR2]][AMR_PARENT]]), &(Bufferrec4_5[n]), &(tempBufferrec4_5[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_PARENT]][45]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_PARENT]][45]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
						MPI_Wait(&boundreqs[n][46], &Statbound[n][46]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive4_6, send2_2, tempreceive4_6, prim,
						&(Bufferp[n]), &(Bufferrec4_6[n]), &(Buffersend2_2[n]), &(tempBufferrec4_6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_6, receive4_6, tempreceive4_6, prim,
						&(Bufferp[n]), &(Buffersend4_6[block[block[n][AMR_NBR2]][AMR_PARENT]]), &(Bufferrec4_6[n]), &(tempBufferrec4_6[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_PARENT]][46]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_PARENT]][46]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
						MPI_Wait(&boundreqs[n][47], &Statbound[n][47]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive4_7, send2_3, tempreceive4_7, prim,
						&(Bufferp[n]), &(Bufferrec4_7[n]), &(Buffersend2_3[n]), &(tempBufferrec4_7[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_7, receive4_7, tempreceive4_7, prim,
						&(Bufferp[n]), &(Buffersend4_7[block[block[n][AMR_NBR2]][AMR_PARENT]]), &(Bufferrec4_7[n]), &(tempBufferrec4_7[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_PARENT]][47]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_PARENT]][47]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
						MPI_Wait(&boundreqs[n][48], &Statbound[n][48]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive4_8, send2_4, tempreceive4_8, prim,
						&(Bufferp[n]), &(Bufferrec4_8[n]), &(Buffersend2_4[n]), &(tempBufferrec4_8[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, N3_GPU[n] + N3G, N2_GPU[n] / (1 + REF_2) + 2 * N2G, (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send4_8, receive4_8, tempreceive4_8, prim,
						&(Bufferp[n]), &(Buffersend4_8[block[block[n][AMR_NBR2]][AMR_PARENT]]), &(Bufferrec4_8[n]), &(tempBufferrec4_8[n]), &(boundevent1[block[block[n][AMR_NBR2]][AMR_PARENT]][48]), &(boundevent2[block[block[n][AMR_NBR2]][AMR_PARENT]][48]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else printf("Error in indexing!\n");
	}
#endif
}

void bound_rec2(double(*restrict prim[NB])[NPR], cl_mem Bufferp[NB], int bound_force, int n){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][30], &Statbound[n][30]);
				}
				if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, 0, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), receive3, tempreceive3, prim,
						&(Bufferp[n]), &(Bufferrec3[n]), &(tempBufferrec3[n]), NULL, NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, 0, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), receive3, tempreceive3, prim,
						&(Bufferp[n]), &(Bufferrec3[n]), &(tempBufferrec3[n]), NULL, NULL, 0);
				}
			}
			else{
				if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, 0, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send1, tempreceive3, prim,
						&(Bufferp[n]), &(Buffersend1[block[n][AMR_NBR1]]), &(tempBufferrec3[n]), &(boundevent1[block[n][AMR_NBR1]][10]), &(boundevent2[block[n][AMR_NBR1]][10]), 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, 0, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send3, tempreceive3, prim,
						&(Bufferp[n]), &(Buffersend3[block[n][AMR_NBR1]]), &(tempBufferrec3[n]), &(boundevent1[block[n][AMR_NBR1]][30]), &(boundevent2[block[n][AMR_NBR1]][30]), 0);
				}
			}
		}
		else if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2){
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
				//receive from finer grid
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][31], &Statbound[n][31]);
					}
					unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive3_1, tempreceive3_1, prim,
						&(Bufferp[n]), &(Bufferrec3_1[n]), &(tempBufferrec3_1[n]), NULL, NULL, 0);
				}
				else{
					unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send3, tempreceive3_1, prim,
						&(Bufferp[n]), &(Buffersend3[block[block[n][AMR_NBR1]][AMR_CHILD3]]), &(tempBufferrec3_1[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_CHILD3]][30]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_CHILD3]][30]), 0);
				}
				if (REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][32], &Statbound[n][32]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive3_2, tempreceive3_2, prim,
							&(Bufferp[n]), &(Bufferrec3_2[n]), &(tempBufferrec3_2[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send3, tempreceive3_2, prim,
							&(Bufferp[n]), &(Buffersend3[block[block[n][AMR_NBR1]][AMR_CHILD4]]), &(tempBufferrec3_2[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_CHILD4]][30]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_CHILD4]][30]), 0);
					}
				}
				if (REF_1 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][35], &Statbound[n][35]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G, 0, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive3_5, tempreceive3_5, prim,
							&(Bufferp[n]), &(Bufferrec3_5[n]), &(tempBufferrec3_5[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G, 0, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send3, tempreceive3_5, prim,
							&(Bufferp[n]), &(Buffersend3[block[block[n][AMR_NBR1]][AMR_CHILD7]]), &(tempBufferrec3_5[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_CHILD7]][30]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_CHILD7]][30]), 0);
					}
				}
				if (REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][36], &Statbound[n][36]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G, 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive3_6, tempreceive3_6, prim,
							&(Bufferp[n]), &(Bufferrec3_6[n]), &(tempBufferrec3_6[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G, 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send3, tempreceive3_6, prim,
							&(Bufferp[n]), &(Buffersend3[block[block[n][AMR_NBR1]][AMR_CHILD8]]), &(tempBufferrec3_6[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_CHILD8]][30]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_CHILD8]][30]), 0);
					}
				}
			}
			else if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//receive from coarser grid
				if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
							MPI_Wait(&boundreqs[n][31], &Statbound[n][31]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive3_1, send1_3, tempreceive3_1, prim,
							&(Bufferp[n]), &(Bufferrec3_1[n]), &(Buffersend1_3[n]), &(tempBufferrec3_1[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_1, receive3_1, tempreceive3_1, prim,
							&(Bufferp[n]), &(Buffersend3_1[block[block[n][AMR_NBR1]][AMR_PARENT]]), &(Bufferrec3_1[n]), &(tempBufferrec3_1[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_PARENT]][31]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_PARENT]][31]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
							MPI_Wait(&boundreqs[n][32], &Statbound[n][32]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive3_2, send1_4, tempreceive3_2, prim,
							&(Bufferp[n]), &(Bufferrec3_2[n]), &(Buffersend1_4[n]), &(tempBufferrec3_2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_2, receive3_2,tempreceive3_2, prim,
							&(Bufferp[n]), &(Buffersend3_2[block[block[n][AMR_NBR1]][AMR_PARENT]]), &(Bufferrec3_2[n]), &(tempBufferrec3_2[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_PARENT]][32]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_PARENT]][32]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
							MPI_Wait(&boundreqs[n][35], &Statbound[n][35]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive3_5, send1_7, tempreceive3_5, prim,
							&(Bufferp[n]), &(Bufferrec3_5[n]), &(Buffersend1_7[n]), &(tempBufferrec3_5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_5, receive3_5, tempreceive3_5, prim,
							&(Bufferp[n]), &(Buffersend3_5[block[block[n][AMR_NBR1]][AMR_PARENT]]), &(Bufferrec3_5[n]), &(tempBufferrec3_5[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_PARENT]][35]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_PARENT]][35]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1)){
							MPI_Wait(&boundreqs[n][36], &Statbound[n][36]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive3_6, send1_8, tempreceive3_6, prim,
							&(Bufferp[n]), &(Bufferrec3_6[n]), &(Buffersend1_8[n]), &(tempBufferrec3_6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, 0,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send3_6, receive3_6, tempreceive3_6, prim,
							&(Bufferp[n]), &(Buffersend3_6[block[block[n][AMR_NBR1]][AMR_PARENT]]), &(Bufferrec3_6[n]), &(tempBufferrec3_6[n]), &(boundevent1[block[block[n][AMR_NBR1]][AMR_PARENT]][36]), &(boundevent2[block[block[n][AMR_NBR1]][AMR_PARENT]][36]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][10], &Statbound[n][10]);
				}
				if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, N1_GPU[n] + N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), receive1, tempreceive1, prim,
						&(Bufferp[n]), &(Bufferrec1[n]), &(tempBufferrec1[n]), NULL, NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, N1_GPU[n] + N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), receive1, tempreceive1, prim,
						&(Bufferp[n]), &(Bufferrec1[n]), &(tempBufferrec1[n]), NULL, NULL, 0);
				}
			}
			else{
				if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, N1_GPU[n] + N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send3, tempreceive1, prim,
						&(Bufferp[n]), &(Buffersend3[block[n][AMR_NBR3]]), &(tempBufferrec1[n]), &(boundevent1[block[n][AMR_NBR3]][30]), &(boundevent2[block[n][AMR_NBR3]][30]), 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, N1_GPU[n] + N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G, N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N3_GPU[n] + 2 * N3G), send1, tempreceive1, prim,
						&(Bufferp[n]), &(Buffersend1[block[n][AMR_NBR3]]), &(tempBufferrec1[n]), &(boundevent1[block[n][AMR_NBR3]][10]), &(boundevent2[block[n][AMR_NBR3]][10]), 0);
				}
			}
		}
		else if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
				//receive from finer grid
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][13], &Statbound[n][13]);
					}
					unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive1_3, tempreceive1_3, prim,
						&(Bufferp[n]), &(Bufferrec1_3[n]), &(tempBufferrec1_3[n]), NULL, NULL, 0);
				}
				else{
					unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send1, tempreceive1_3, prim,
						&(Bufferp[n]), &(Buffersend1[block[block[n][AMR_NBR3]][AMR_CHILD1]]), &(tempBufferrec1_3[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_CHILD1]][10]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_CHILD1]][10]), 0);
				}
				if (REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][14], &Statbound[n][14]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive1_4, tempreceive1_4, prim,
							&(Bufferp[n]), &(Bufferrec1_4[n]), &(tempBufferrec1_4[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, N2_GPU[n], N2_GPU[n] + N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send1, tempreceive1_4, prim,
							&(Bufferp[n]), &(Buffersend1[block[block[n][AMR_NBR3]][AMR_CHILD2]]), &(tempBufferrec1_4[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_CHILD2]][10]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_CHILD2]][10]), 0);
					}
				}
				if (REF_1 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][17], &Statbound[n][17]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive1_7, tempreceive1_7, prim,
							&(Bufferp[n]), &(Bufferrec1_7[n]), &(tempBufferrec1_7[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, N2_GPU[n], N2_GPU[n] + N2G, 0, -N3G / (1 + REF_3), N3_GPU[n] / (1 + REF_3) + (1 - REF_3)*N3G,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send1, tempreceive1_7, prim,
							&(Bufferp[n]), &(Buffersend1[block[block[n][AMR_NBR3]][AMR_CHILD5]]), &(tempBufferrec1_7[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_CHILD5]][10]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_CHILD5]][10]), 0);
					}
				}
				if (REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][18], &Statbound[n][18]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, N2_GPU[n], N2_GPU[n] + N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), receive1_8, tempreceive1_8, prim,
							&(Bufferp[n]), &(Bufferrec1_8[n]), &(tempBufferrec1_8[n]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, N2_GPU[n], N2_GPU[n] + N2G, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n] + D3*REF_3,
							(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N3_GPU[n] + 2 * N3G) / (1 + REF_3), send1, tempreceive1_8, prim,
							&(Bufferp[n]), &(Buffersend1[block[block[n][AMR_NBR3]][AMR_CHILD6]]), &(tempBufferrec1_8[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_CHILD6]][10]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_CHILD6]][10]), 0);
					}
				}
			}
			else if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//receive from coarser grid
				if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][13], &Statbound[n][13]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive1_3, send3_1, tempreceive1_3, prim,
							&(Bufferp[n]), &(Bufferrec1_3[n]), &(Buffersend3_1[n]), &(tempBufferrec1_3[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_3, receive1_3, tempreceive1_3, prim,
							&(Bufferp[n]), &(Buffersend1_3[block[block[n][AMR_NBR3]][AMR_PARENT]]), &(Bufferrec1_3[n]), &(tempBufferrec1_3[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_PARENT]][13]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_PARENT]][13]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][14], &Statbound[n][14]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive1_4, send3_2, tempreceive1_4, prim,
							&(Bufferp[n]), &(Bufferrec1_4[n]), &(Buffersend3_2[n]), &(tempBufferrec1_4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_4, receive1_4, tempreceive1_4, prim,
							&(Bufferp[n]), &(Buffersend1_4[block[block[n][AMR_NBR3]][AMR_PARENT]]), &(Bufferrec1_4[n]), &(tempBufferrec1_4[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_PARENT]][14]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_PARENT]][14]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][17], &Statbound[n][17]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive1_7, send3_5, tempreceive1_7, prim,
							&(Bufferp[n]), &(Bufferrec1_7[n]), &(Buffersend3_5[n]), &(tempBufferrec1_7[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_7, receive1_7, tempreceive1_7, prim,
							&(Bufferp[n]), &(Buffersend1_7[block[block[n][AMR_NBR3]][AMR_PARENT]]), &(Bufferrec1_7[n]), &(tempBufferrec1_7[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_PARENT]][17]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_PARENT]][17]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
							MPI_Wait(&boundreqs[n][18], &Statbound[n][18]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), receive1_8, send3_6, tempreceive1_8, prim,
							&(Bufferp[n]), &(Bufferrec1_8[n]), &(Buffersend3_6[n]), &(tempBufferrec1_8[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + N2G,
							-N3G, N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N3_GPU[n] / (1 + REF_3) + 2 * N3G), send1_8, receive1_8, tempreceive1_8, prim,
							&(Bufferp[n]), &(Buffersend1_8[block[block[n][AMR_NBR3]][AMR_PARENT]]), &(Bufferrec1_8[n]), &(tempBufferrec1_8[n]), &(boundevent1[block[block[n][AMR_NBR3]][AMR_PARENT]][18]), &(boundevent2[block[block[n][AMR_NBR3]][AMR_PARENT]][18]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
			}
		}
	}
#endif
}

void bound_rec3(double(*restrict prim[NB])[NPR], cl_mem Bufferp[NB], int bound_force, int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][50], &Statbound[n][50]);
				}
				unpack_receive3(n, block[n][AMR_NBR6], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, 0, (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), receive5, tempreceive5, prim,
					&(Bufferp[n]), &(Bufferrec5[n]), &(tempBufferrec5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[n][AMR_NBR6], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, -N3G, 0, (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), send5, tempreceive5, prim,
					&(Bufferp[n]), &(Buffersend5[block[n][AMR_NBR6]]), &(tempBufferrec5[n]), &(boundevent1[block[n][AMR_NBR6]][50]), &(boundevent2[block[n][AMR_NBR6]][50]), block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][51], &Statbound[n][51]);
				}
				unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive5_1, tempreceive5_1, prim,
					&(Bufferp[n]), &(Bufferrec5_1[n]), &(tempBufferrec5_1[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send5, tempreceive5_1, prim,
					&(Bufferp[n]), &(Buffersend5[block[block[n][AMR_NBR6]][AMR_CHILD2]]), &(tempBufferrec5_1[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_CHILD2]][50]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_CHILD2]][50]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][53], &Statbound[n][53]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive5_3, tempreceive5_3, prim,
						&(Bufferp[n]), &(Bufferrec5_3[n]), &(tempBufferrec5_3[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send5, tempreceive5_3, prim,
						&(Bufferp[n]), &(Buffersend5[block[block[n][AMR_NBR6]][AMR_CHILD4]]), &(tempBufferrec5_3[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_CHILD4]][50]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_CHILD4]][50]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][55], &Statbound[n][55]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive5_5, tempreceive5_5, prim,
						&(Bufferp[n]), &(Bufferrec5_5[n]), &(tempBufferrec5_5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send5, tempreceive5_5, prim,
						&(Bufferp[n]), &(Buffersend5[block[block[n][AMR_NBR6]][AMR_CHILD6]]), &(tempBufferrec5_5[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_CHILD6]][50]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_CHILD6]][50]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][57], &Statbound[n][57]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive5_7, tempreceive5_7, prim,
						&(Bufferp[n]), &(Bufferrec5_7[n]), &(tempBufferrec5_7[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, -N3G, 0,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send5, tempreceive5_7, prim,
						&(Bufferp[n]), &(Buffersend5[block[block[n][AMR_NBR6]][AMR_CHILD8]]), &(tempBufferrec5_7[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_CHILD8]][50]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_CHILD8]][50]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][51], &Statbound[n][51]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive5_1, send6_2, tempreceive5_1, prim,
						&(Bufferp[n]), &(Bufferrec5_1[n]), &(Buffersend6_2[n]), &(tempBufferrec5_1[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_1, receive5_1, tempreceive5_1, prim,
						&(Bufferp[n]), &(Buffersend5_1[block[block[n][AMR_NBR6]][AMR_PARENT]]), &(Bufferrec5_1[n]), &(tempBufferrec5_1[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_PARENT]][51]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_PARENT]][51]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][53], &Statbound[n][53]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive5_3, send6_4, tempreceive5_3, prim,
						&(Bufferp[n]), &(Bufferrec5_3[n]), &(Buffersend6_4[n]), &(tempBufferrec5_3[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_3, receive5_3, tempreceive5_3, prim,
						&(Bufferp[n]), &(Buffersend5_3[block[block[n][AMR_NBR6]][AMR_PARENT]]), &(Bufferrec5_3[n]), &(tempBufferrec5_3[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_PARENT]][53]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_PARENT]][53]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][55], &Statbound[n][55]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive5_5, send6_6, tempreceive5_5, prim,
						&(Bufferp[n]), &(Bufferrec5_5[n]), &(Buffersend6_6[n]), &(tempBufferrec5_5[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_5, receive5_5, tempreceive5_5, prim,
						&(Bufferp[n]), &(Buffersend5_5[block[block[n][AMR_NBR6]][AMR_PARENT]]), &(Bufferrec5_5[n]), &(tempBufferrec5_5[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_PARENT]][55]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_PARENT]][55]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][57], &Statbound[n][57]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive5_7, send6_8, tempreceive5_7, prim,
						&(Bufferp[n]), &(Bufferrec5_7[n]), &(Buffersend6_8[n]), &(tempBufferrec5_7[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						-N3G, 0, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send5_7, receive5_7, tempreceive5_7, prim,
						&(Bufferp[n]), &(Buffersend5_7[block[block[n][AMR_NBR6]][AMR_PARENT]]), &(Bufferrec5_7[n]), &(tempBufferrec5_7[n]), &(boundevent1[block[block[n][AMR_NBR6]][AMR_PARENT]][57]), &(boundevent2[block[block[n][AMR_NBR6]][AMR_PARENT]][57]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else printf("Error in indexing!\n");
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][60], &Statbound[n][60]);
				}
				unpack_receive3(n, block[n][AMR_NBR5], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), receive6, tempreceive6, prim,
					&(Bufferp[n]), &(Bufferrec6[n]), &(tempBufferrec6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[n][AMR_NBR5], 0, -N1G, N1_GPU[n] + N1G, 0, -N2G, N2_GPU[n] + N2G, 0, N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] + 2 * N1G), (N2_GPU[n] + 2 * N2G), send6, tempreceive6, prim,
					&(Bufferp[n]), &(Buffersend6[block[n][AMR_NBR5]]), &(tempBufferrec6[n]), &(boundevent1[block[n][AMR_NBR5]][60]), &(boundevent2[block[n][AMR_NBR5]][60]), block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1){
					MPI_Wait(&boundreqs[n][62], &Statbound[n][62]);
				}
				unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, N3_GPU[n], N3_GPU[n] + N3G,
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive6_2, tempreceive6_2, prim,
					&(Bufferp[n]), &(Bufferrec6_2[n]), &(tempBufferrec6_2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, N3_GPU[n], N3_GPU[n] + N3G,
					(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send6, tempreceive6_2, prim,
					&(Bufferp[n]), &(Buffersend6[block[block[n][AMR_NBR5]][AMR_CHILD1]]), &(tempBufferrec6_2[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_CHILD1]][60]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_CHILD1]][60]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][64], &Statbound[n][64]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive6_4, tempreceive6_4, prim,
						&(Bufferp[n]), &(Bufferrec6_4[n]), &(tempBufferrec6_4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, -N1G / (1 + REF_1), N1_GPU[n] / (1 + REF_1) + (1 - REF_1)*N1G, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send6, tempreceive6_4, prim,
						&(Bufferp[n]), &(Buffersend6[block[block[n][AMR_NBR5]][AMR_CHILD3]]), &(tempBufferrec6_4[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_CHILD3]][60]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_CHILD3]][60]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][66], &Statbound[n][66]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive6_6, tempreceive6_6, prim,
						&(Bufferp[n]), &(Bufferrec6_6[n]), &(tempBufferrec6_6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 0, -N2G / (1 + REF_2), N2_GPU[n] / (1 + REF_2) + (1 - REF_2)*N2G, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send6, tempreceive6_6, prim,
						&(Bufferp[n]), &(Buffersend6[block[block[n][AMR_NBR5]][AMR_CHILD5]]), &(tempBufferrec6_6[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_CHILD5]][60]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_CHILD5]][60]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][68], &Statbound[n][68]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), receive6_8, tempreceive6_8, prim,
						&(Bufferp[n]), &(Bufferrec6_8[n]), &(tempBufferrec6_8[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], 1, N1_GPU[n] / (1 + REF_1), N1_GPU[n] + D1*REF_1, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n] + D2*REF_2, 0, N3_GPU[n], N3_GPU[n] + N3G,
						(N1_GPU[n] + 2 * N1G) / (1 + REF_1), (N2_GPU[n] + 2 * N2G) / (1 + REF_2), send6, tempreceive6_8, prim,
						&(Bufferp[n]), &(Buffersend6[block[block[n][AMR_NBR5]][AMR_CHILD7]]), &(tempBufferrec6_8[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_CHILD7]][60]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_CHILD7]][60]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][62], &Statbound[n][62]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive6_2, send5_1, tempreceive6_2, prim,
						&(Bufferp[n]), &(Bufferrec6_2[n]), &(Buffersend5_1[n]), &(tempBufferrec6_2[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_2, receive6_2, tempreceive6_2, prim,
						&(Bufferp[n]), &(Buffersend6_2[block[block[n][AMR_NBR5]][AMR_PARENT]]), &(Bufferrec6_2[n]), &(tempBufferrec6_2[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_PARENT]][62]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_PARENT]][62]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][64], &Statbound[n][64]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive6_4, send5_3, tempreceive6_4, prim,
						&(Bufferp[n]), &(Bufferrec6_4[n]), &(Buffersend5_3[n]), &(tempBufferrec6_4[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_4, receive6_4, tempreceive6_4, prim,
						&(Bufferp[n]), &(Buffersend6_4[block[block[n][AMR_NBR5]][AMR_PARENT]]), &(Bufferrec6_4[n]), &(tempBufferrec6_4[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_PARENT]][64]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_PARENT]][64]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][66], &Statbound[n][66]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive6_6, send5_5, tempreceive6_6, prim,
						&(Bufferp[n]), &(Bufferrec6_6[n]), &(Buffersend5_5[n]), &(tempBufferrec6_6[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_6, receive6_6, tempreceive6_6, prim,
						&(Bufferp[n]), &(Buffersend6_6[block[block[n][AMR_NBR5]][AMR_PARENT]]), &(Bufferrec6_6[n]), &(tempBufferrec6_6[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_PARENT]][66]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_PARENT]][66]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1){
						MPI_Wait(&boundreqs[n][68], &Statbound[n][68]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), receive6_8, send5_7, tempreceive6_8, prim,
						&(Bufferp[n]), &(Bufferrec6_8[n]), &(Buffersend5_7[n]), &(tempBufferrec6_8[n]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, N1_GPU[n] + N1G, -N2G, N2_GPU[n] + N2G,
						N3_GPU[n], N3_GPU[n] + N3G, (N1_GPU[n] / (1 + REF_1) + 2 * N1G), (N2_GPU[n] / (1 + REF_2) + 2 * N2G), send6_8, receive6_8, tempreceive6_8, prim,
						&(Bufferp[n]), &(Buffersend6_8[block[block[n][AMR_NBR5]][AMR_PARENT]]), &(Bufferrec6_8[n]), &(tempBufferrec6_8[n]), &(boundevent1[block[block[n][AMR_NBR5]][AMR_PARENT]][68]), &(boundevent2[block[block[n][AMR_NBR5]][AMR_PARENT]][68]), block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
	}
#endif
}