#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}
void pack_send1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1)*(z2 - z1);
		 packsend1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		 if(block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send1 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + k] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (0 + NPR)] =
						ps[nl[n]][index_3D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[nl[n]][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (1 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[nl[n]][(NPR + 3)*(i - i1)*zsize*jsize + (NPR + 3)*(j - j1)*zsize + (NPR + 3)*(z - z1) + (2 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
#endif
				}
			}
		}
	}
}

void pack_send2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(z2 - z1);
		 packsend2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		 if (block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 //cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send2 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (j = j1; j < j2; j++){
			for (i = i1; i < i2; i++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + k] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (0 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[nl[n]][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (1 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[nl[n]][(NPR + 3)*(j - j1)*zsize*isize + (NPR + 3)*(i - i1)*zsize + (NPR + 3)*(z - z1) + (2 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];

#endif
				}
			}
		}
	}
}

void pack_send3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		 packsend3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		 if (block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 //cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send3 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + k] = prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (0 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					send[nl[n]][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (1 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					send[nl[n]][(NPR + 3)*(z - z1)*jsize*isize + (NPR + 3)*(i - i1)*jsize + (NPR + 3)*(j - j1) + (2 + NPR)] =
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1>N3G) + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1>N3G) + N3_GPU_offset[n])][FACE3];
#endif
				}
			}
		}
	}
}

void pack_send_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3);
		 packsendaverage1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		 if (block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 //cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send_average1 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (i = i1; i < i2; i += 1 + REF_1){
			for (j = j1; j < j2; j += 1 + REF_2){
				for (z = z1; z < z2; z += (1 + REF_3)){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + k]
							= 0.125*(prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (0 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						ps[nl[n]][index_3D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE1]
						+ ps[nl[n]][index_3D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE1] +
						ps[nl[n]][index_3D(n, i + (i1>N1G) + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][FACE1]);

					send[nl[n]][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (1 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]);

					send[nl[n]][(NPR + 3)*(i - i1) / (1 + REF_1)*zsize*jsize + (NPR + 3)*(j - j1) / (1 + REF_2)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (2 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]);
#endif
				}
			}
		}
	}
}

void pack_send_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3);
		 packsendaverage2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		 if (block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 //cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send_average2 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (j = j1; j < j2; j += 1 + REF_2){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (z = z1; z < z2; z += 1 + REF_3){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + k]
							= 0.125*(prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (0 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE1]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE1] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][FACE1]);

					send[nl[n]][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (1 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + (j1>N2G) + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]);

					send[nl[n]][(NPR + 3)*(j - j1) / (1 + REF_2)*isize*zsize + (NPR + 3)*(i - i1) / (1 + REF_1)*zsize + (NPR + 3)*(z - z1) / (1 + REF_3) + (2 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] +
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]);
#endif
				}
			}
		}
	}
}

void pack_send_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + REF_1)*(j2 - j1) / (1 + REF_2)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + REF_1)*(j2 - j1) / (1 + REF_2);
		packsendaverage3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferps[0], Bufferboundsend[0], Buffergdet[nl[n]], work_size);
		if (block[n_rec][AMR_NODE] == block[n][AMR_NODE]) cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error pack_send_average3 %d", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (z = z1; z < z2; z += 1 + REF_3){
			for (i = i1; i < i2; i += 1 + REF_1){
				for (j = j1; j < j2; j += 1 + REF_2){
					for (k = 0; k < NPR; k++){
						send[nl[n]][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + k]
							= 0.125*(prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
							+ prim[nl[n]][index_3D(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
					}
#if(STAGGERED)
					send[nl[n]][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (0 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE1]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE1]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][1] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][FACE1]);
					send[nl[n]][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (1 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][2] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][FACE2]);
					send[nl[n]][(NPR + 3)*(z - z1) / (1 + REF_3)*isize*jsize + (NPR + 3)*(i - i1) / (1 + REF_1)*jsize + (NPR + 3)*(j - j1) / (1 + REF_2) + (2 + NPR)] =
						0.25*(ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1>N3G) + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + (z1>N3G) + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + (z1>N3G) + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3]
						+ ps[nl[n]][index_3D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + (z1>N3G) + N3_GPU_offset[n])][3] * gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][FACE3]);
#endif
				}
			}
		}
	}
}

void unpack_receive1(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE]==block[n_rec][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		int work_size = (j2 - j1)*(z2 - z1);
		 unpackreceive1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, jsize, zsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferboundreceive[0], tempBufferboundreceive[0],
			update_staggered, Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "unpack_receive1 error! %d \n", status);
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
						ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
					}
#if(STAGGERED)
					if (update_staggered == 1 && (z<0 || z >= BS_3 || j<0 || j >= BS_2)){
						ps[nl[n]][index_3D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)] /
							gdet[nl[n]][index_2D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						psh[nl[n]][index_3D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)] /
							gdet[nl[n]][index_2D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					}
					else psh[nl[n]][index_3D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)] /
						gdet[nl[n]][index_2D(n, i + (i1<0) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
					psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
#endif
				}
			}
		}
	}
}

void unpack_receive2(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int isize, int zsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int reverse){
	double factor = 1.;
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE] == block[n_rec][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		int work_size = (i2 - i1)*(z2 - z1);
		 unpackreceive2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, isize, zsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferboundreceive[0], tempBufferboundreceive[0],
			reverse, update_staggered, Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "unpack_receive2 error! \n");
	}
	else{
		int i, j, z, k;
		if (reverse == 0){
			#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						for (k = 0; k < NPR; k++){
							p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
							ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
						}
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
						if (update_staggered == 1 && (i<0 || i >= BS_1 || z<0 || z >= BS_3)){
							ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)] /
								gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
							psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)] /
								gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
						}
						else psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + (j1<0) + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
						psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2))*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
#endif
					}
				}
			}
		}
		else{
			#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,factor)
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				for (k = 0; k < NPR; k++){
					factor=1.;
					if (k == U3 || k == U2 || k == B2 || k == B3) factor = -1.;
					else factor = 1.;
					p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = factor*receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
					ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = factor*receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + k];
				}
				/*double ucon[NDIM], utcon[NDIM];
				struct of_geom geom;

				if (j<0)get_geometry(n, i + N1_GPU_offset[n], -j-1 + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				else get_geometry(n, i + N1_GPU_offset[n], 2*BS_2-j-1 + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				get_geometry(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);
				ucon_calc(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])], &geom, ucon);
				ucon[3] *= -1.;
				ucon[2] *= -1.;

				//get_geometry(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n], CENT, &geom);

				ucon_to_utcon(ucon, &geom, utcon);
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U2] = utcon[2];
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U3] = utcon[3];
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U2] = utcon[2];
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][U3] = utcon[3];

				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B2] *= -1.;
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B3] *= -1.;
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B2] *= -1.;
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][B3] *= -1.;
				*/
#if(STAGGERED)
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)];
				if (nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || nstep == -1){
					//ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = -receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)];
				}
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = -receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (0 + NPR)];
				//psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j+ N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = -receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (1 + NPR)];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = -receive[nl[n_rec2]][(NPR + 3)*(j2 - 1 - j)*zsize*isize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*zsize + (NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3)) + (2 + NPR)];
#endif
			}
		}
	}
}

void unpack_receive3(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int isize, int jsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	int update_staggered = ((nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1) || nstep == -1);
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE] == block[n_rec][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, i_offset, j1, j2, j_offset, z1, z2, z_offset, isize, jsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferboundreceive[0], tempBufferboundreceive[0],
			update_staggered, Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "unpack_receive3 error! \n");
	}
	else{
		int i, j, z, k;
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k)
		for (z = z1; z < z2; z++){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (k = 0; k < NPR; k++){
						p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + k];
						ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + k];
					}
#if(STAGGERED)
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (0 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (0 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (1 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (1 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
					if (update_staggered == 1 && (i<0 || i >= BS_1 || j<0 || j >= BS_2)){
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (2 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][FACE3];
						psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (2 + NPR)] /
							gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][FACE3];
					}
					else psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][3] = receive[nl[n_rec2]][(NPR + 3)*(z - z1 + z_offset * 2 * D3 / (1 + REF_3))*isize*jsize + (NPR + 3)*(i - i1 + i_offset * 2 * D1 / (1 + REF_1))*jsize + (NPR + 3)*(j - j1 + j_offset * 2 * D2 / (1 + REF_2)) + (2 + NPR)] /
						gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + (z1<0) + N3_GPU_offset[n])][FACE3];
#endif
				}
			}
		}
	}
}

void unpack_receive_coarse1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE] == block[n_rec][AMR_NODE])cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		int work_size = (j2 - j1)*(z2 - z1);
		unpackreceivecoarse1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferp[0], Bufferps[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 ////cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error unpack_receive_coarse1: %d \n", status);
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1[NPR + NDIM], dq2[NPR + NDIM], dq3[NPR + NDIM], avg[NPR + NDIM];
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,ii, ij, iz,is, js, zs, dq1, dq2, dq3, avg)
		for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
			//Use slope limited interpolation in direction fluxes, copy  boundary cells in other directions
			if (i1 < 0 && REF_1 == 1) ii = NG - 1;
			else if (REF_1 == 1)ii = 0;
			else ii = i - i1;
			ij = (j - j1 - (j - j1) % (1 + REF_2)) / (1 + REF_2) + (NG - 1)*REF_2;
			iz = (z - z1 - (z - z1) % (1 + REF_3)) / (1 + REF_3) + (NG - 1)*REF_3;

			is = ((i == i1) ? (-1) : (1));
			js = (((j - j1) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = (((z - z1) % (1 + REF_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR; k++){
				avg[k] = 0.125*(prim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n])][k] + prim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n])][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k] + prim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, z - z % (1 + REF_3) + N3_GPU_offset[n])][k] + prim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, z - z % (1 + REF_3) + N3_GPU_offset[n])][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k] + prim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k]);

				if (ii == 0){
					dq1[k] = slope_lim(avg[k], receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+k], receive[nl[n_rec2]][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + k]);
				}
				else{
					dq1[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+k], receive[nl[n_rec2]][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + k], avg[k]);
				}
				dq2[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij - REF_2)*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + k]);
				dq3[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz - REF_3) + k], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz - REF_3) + k]);
			}
			for (k = 0; k < 3; k++){
				avg[k + NPR] = 0.25*(psim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n])][k + 1] + psim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n], z - z % (1 + REF_3) + N3_GPU_offset[n])][k + 1])*
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];
				avg[k + NPR] += 0.25*(psim[nl[n]][index_3D(n, N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2*(k == 2), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3*(k == 1))][k + 1] + psim[nl[n]][index_3D(n, ii / (NG - 1) + N1_GPU_offset[n] + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2*(k == 2), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3*(k == 1))][k + 1])*
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];
				if (ii == 0){
					dq1[k + NPR] = slope_lim(avg[k + NPR], receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (k + NPR)]);
				}
				else{
					dq1[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_1*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (k + NPR)], avg[k + NPR]);
				}
				dq2[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij - REF_2)*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + (k + NPR)]);
				dq3[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz - REF_3) + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*(iz - REF_3) + (k + NPR)]);
			}

			for (k = 0; k < NPR; k++){
				//dq2[k] = dq3[k] = 0.;
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (k)] + 0.25*(double)(is)*dq1[(k)] * REF_1 + 0.25*(double)(js)*dq2[(k)] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (k)] + 0.25*(double)(is)*dq1[(k)] * REF_1 + 0.25*(double)(js)*dq2[(k)] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
			}

			#if(STAGGERED)
			if (js == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (1 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + (1 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] + 0.25*(double)(is)*dq1[B2] * REF_1 + 0.25*(double)(zs)*dq3[B2] * REF_3;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij)*zsize + (NPR + 3)*iz + (1 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*(ij + REF_2)*zsize + (NPR + 3)*iz + (1 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] + 0.25*(double)(is)*dq1[B2] * REF_1 + 0.25*(double)(zs)*dq3[B2] * REF_3;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = (receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (1 + NPR)] + 0.25*(double)(is)*dq1[NPR + 1] * REF_1 + 0.25*(double)(zs)*dq3[NPR + 1] * REF_3) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = (receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (1 + NPR)] + 0.25*(double)(is)*dq1[NPR + 1] * REF_1 + 0.25*(double)(zs)*dq3[NPR + 1] * REF_3) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			if (zs == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] + 0.25*(double)(is)*dq1[B3] * REF_1 + 0.25*(double)(js)*dq2[B3] * REF_2;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] + 0.25*(double)(is)*dq1[B3] * REF_1 + 0.25*(double)(js)*dq2[B3] * REF_2;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = (receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)] + 0.25*(double)(is)*dq1[NPR + 2] * REF_1 + 0.25*(double)(js)*dq2[NPR + 2] * REF_2) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = (receive[nl[n_rec2]][(NPR + 3)*ii*zsize*jsize + (NPR + 3)*ij*zsize + (NPR + 3)*iz + (2 + NPR)] + 0.25*(double)(is)*dq1[NPR + 2] * REF_1 + 0.25*(double)(js)*dq2[NPR + 2] * REF_2) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
#endif
		}
	}
}

void unpack_receive_coarse2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE] == block[n_rec][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		int work_size = (i2 - i1)*(z2 - z1);
		unpackreceivecoarse2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferp[0], Bufferps[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error unpack_receive_coarse2");
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1[NPR + NDIM], dq2[NPR + NDIM], dq3[NPR + NDIM], avg[NPR + NDIM];
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,ii, ij, iz,is, js, zs, dq1, dq2, dq3, avg)
		for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
			//now use zero order interpolation, must be done better in the future
			if (j1 < 0 && REF_2 == 1) ij = NG - 1;
			else if (REF_2 == 1) ij = 0;
			else ij = j - j1;
			ii = (i - i1 - (i - i1) % (1 + REF_1)) / (1 + REF_1) + (NG - 1)*REF_1;
			iz = (z - z1 - (z - z1) % (1 + REF_3)) / (1 + REF_3) + (NG - 1)*REF_3;

			is = (((i - i1) % (1 + REF_1) == 0) ? (-1) : (1));
			js = ((j == j1) ? (-1) : (1));
			zs = (((z - z1) % (1 + REF_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR; k++){
				avg[k] = 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n])][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n])][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n])][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n])][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3)][k]);

				if (ij == 0){
					dq2[k] = slope_lim(avg[k], receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+k], receive[nl[n_rec2]][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + k]);
				}
				else{
					dq2[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+k], receive[nl[n_rec2]][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + k], avg[k]);
				}
				dq1[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii - REF_1)*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii + REF_1)*zsize + (NPR + 3)*iz + k]);
				dq3[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz - REF_3) + k], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz + REF_3) + k]);
			}
			for (k = 0; k < 3; k++){
				avg[k + NPR] = 0.25*(psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n])][k + 1] + psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n])][k + 1])*
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];
				avg[k + NPR] += 0.25*(psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1*(k == 2), N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3*(k == 0))][k + 1] + psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1*(k == 2), ij / (NG - 1) + N2_GPU_offset[n] + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2), z - z % (1 + REF_3) + N3_GPU_offset[n] + REF_3*(k == 0))][k + 1])*
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];

				if (ij == 0){
					dq2[k + NPR] = slope_lim(avg[k + NPR], receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + (k + NPR)]);
				}
				else{
					dq2[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_2*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*iz + (k + NPR)], avg[k + NPR]);
				}
				dq1[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii - REF_1)*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii + REF_1)*zsize + (NPR + 3)*iz + (k + NPR)]);
				dq3[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz - REF_3) + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*(ii)*zsize + (NPR + 3)*(iz + REF_3) + (k + NPR)]);
			}
			for (k = 0; k < NPR; k++){
				//dq1[k] = dq3[k] = 0.;
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1[k] * REF_1 + 0.25*(double)(js)*dq2[k] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + k] + 0.25*(double)(is)*dq1[k] * REF_1 + 0.25*(double)(js)*dq2[k] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
			}
			#if(STAGGERED)
			if (is == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*(ij)*zsize*isize + (NPR + 3)*(ii + REF_1)*zsize + (NPR + 3)*iz + (0 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] + 0.25*(double)(js)*dq2[B1] * REF_2 + 0.25*(double)(zs)*dq3[B1] * REF_3;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*(ij)*zsize*isize + (NPR + 3)*(ii + REF_1)*zsize + (NPR + 3)*iz + (0 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] + 0.25*(double)(js)*dq2[B1] * REF_2 + 0.25*(double)(zs)*dq3[B1] * REF_3;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = (receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + 0.25*(double)(js)*dq2[NPR] * REF_2 + 0.25*(double)(zs)*dq3[NPR] * REF_3) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = (receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (0 + NPR)] + 0.25*(double)(js)*dq2[NPR] * REF_2 + 0.25*(double)(zs)*dq3[NPR] * REF_3) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			if (zs == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] + 0.25*(double)(js)*dq2[B3] * REF_2 + 0.25*(double)(is)*dq1[B3] * REF_1;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*(iz + REF_3) + (2 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] + 0.25*(double)(js)*dq2[B3] * REF_2 + 0.25*(double)(is)*dq1[B3] * REF_1;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = (receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + 0.25*(double)(js)*dq2[NPR + 2] * REF_2 + 0.25*(double)(is)*dq1[NPR + 2] * REF_1) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = (receive[nl[n_rec2]][(NPR + 3)*ij*zsize*isize + (NPR + 3)*ii*zsize + (NPR + 3)*iz + (2 + NPR)] + 0.25*(double)(js)*dq2[NPR + 2] * REF_2 + 0.25*(double)(is)*dq1[NPR + 2] * REF_1) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3];
			}
			#endif
		}
	}
}

void unpack_receive_coarse3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent, cudaEvent_t *boundevent2, int mpi){
	int n_rec2 = n;
	if (block[n_rec][AMR_NODE] == rank) n_rec2 = n_rec;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		if (nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n_rec][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && nstep % (block[n_rec][AMR_TIMELEVEL]) == block[n][AMR_TIMELEVEL] - 1)){
			if (block[n][AMR_NODE] == block[n_rec][AMR_NODE])cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		int work_size = (i2 - i1)*(j2 - j1);
		unpackreceivecoarse3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], Bufferp[0], Bufferps[0], Bufferboundreceive[0],
			temp1Bufferboundreceive[0], temp2Bufferboundreceive[0], Buffergdet[nl[n]], nstep, dt, block[n][AMR_TIMELEVEL], block[n_rec][AMR_TIMELEVEL], work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess)fprintf(stderr, "error unpack_receive_coarse3");
	}
	else{
		int i, j, z, k;
		int ii, ij, iz;
		int is, js, zs;
		double dq1[NPR + NDIM], dq2[NPR + NDIM], dq3[NPR + NDIM], avg[NPR + NDIM];
		#pragma omp parallel for schedule(dynamic,1) private(i,j,z,k,ii, ij, iz,is, js, zs, dq1, dq2, dq3, avg)
		for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
			//now use zero order interpolation, must be done better in the future
			if (z1 < 0 && REF_3 == 1) iz = NG - 1;
			else if (REF_3 == 1) iz = 0;
			else iz = z - z1;
			ij = (j - j1 - (j - j1) % (1 + REF_2)) / (1 + REF_2) + (NG - 1)*REF_2;
			ii = (i - i1 - (i - i1) % (1 + REF_1)) / (1 + REF_1) + (NG - 1)*REF_1;

			is = (((i - i1) % (1 + REF_1) == 0) ? (-1) : (1));
			js = (((j - j1) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = ((z == z1) ? (-1) : (1));

			for (k = 0; k < NPR; k++){
				avg[k] = 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n], N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n], iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, j - j % (1 + REF_2) + N2_GPU_offset[n], N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, j - j % (1 + REF_2) + N2_GPU_offset[n], iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k]);
				avg[k] += 0.125*(prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k] + prim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1, j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2, iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k]);

				if (iz == 0){
					dq3[k] = slope_lim(avg[k], receive[nl[n_rec2]][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+k], receive[nl[n_rec2]][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + k]);
				}
				else{
					dq3[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+k], receive[nl[n_rec2]][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + k], avg[k]);
				}
				dq1[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii - REF_1)*jsize + (NPR + 3)*ij + k], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii + REF_1)*jsize + (NPR + 3)*ij + k]);
				dq2[k] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij - REF_2) + k], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij + REF_2) + k]);
			}
			for (k = 0; k < 3; k++){
				avg[k + NPR] = 0.25*(psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n], N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k + 1] + psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j - j % (1 + REF_2) + N2_GPU_offset[n], iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k + 1])*
					gdet[nl[n]][index_2D(n, i - i % (1 + REF_1) + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];
				avg[k + NPR] += 0.25*(psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1*(k == 1), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2*(k == 0), N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))][k + 1] + psim[nl[n]][index_3D(n, i - i % (1 + REF_1) + N1_GPU_offset[n] + REF_1*(k == 1), j - j % (1 + REF_2) + N2_GPU_offset[n] + REF_2*(k == 0), iz / (NG - 1) + N3_GPU_offset[n] + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))][k + 1])*
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k + 2 * (k == 2)];

				if (iz == 0){
					dq3[k + NPR] = slope_lim(avg[k + NPR], receive[nl[n_rec2]][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + (k + NPR)]);
				}
				else{
					dq3[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3) * 0 * jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij)+(k + NPR)], receive[nl[n_rec2]][(NPR + 3)*REF_3*jsize*isize + (NPR + 3)*(ii)*jsize + (NPR + 3)*ij + (k + NPR)], avg[k + NPR]);
				}
				dq1[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii - REF_1)*jsize + (NPR + 3)*ij + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii + REF_1)*jsize + (NPR + 3)*ij + (k + NPR)]);
				dq2[k + NPR] = slope_lim(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij - REF_2) + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (k + NPR)], receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii)*jsize + (NPR + 3)*(ij + REF_2) + (k + NPR)]);
			}
			for (k = 0; k < NPR; k++){
				//dq1[k] = dq2[k] = 0.;
				ph[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k] + 0.25*(double)(is)*dq1[k] * REF_1 + 0.25*(double)(js)*dq2[k] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
				p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + k] + 0.25*(double)(is)*dq1[k] * REF_1 + 0.25*(double)(js)*dq2[k] * REF_2 + 0.25*(double)(zs)*dq3[k] * REF_3;
			}
			#if(STAGGERED)
			if (is == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii + REF_1)*jsize + (NPR + 3)*ij + (0 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] + 0.25*(double)(zs)*dq3[B1] * REF_3 + 0.25*(double)(js)*dq2[B1] * REF_2;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*(ii + REF_1)*jsize + (NPR + 3)*ij + (0 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] + 0.25*(double)(zs)*dq3[B1] * REF_3 + 0.25*(double)(js)*dq2[B1] * REF_2;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = (receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + 0.25*(double)(zs)*dq3[NPR + 0] * REF_3 + 0.25*(double)(js)*dq2[NPR + 0] * REF_2) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = (receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (0 + NPR)] + 0.25*(double)(zs)*dq3[NPR + 0] * REF_3 + 0.25*(double)(js)*dq2[NPR + 0] * REF_2) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1];
			}
			if (js == 1){
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*(ij + REF_2) + (1 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] + 0.25*(double)(zs)*dq3[B2] * REF_3 + 0.25*(double)(is)*dq1[B2] * REF_1;
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = 0.5*(receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*(ij + REF_2) + (1 + NPR)]) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] + 0.25*(double)(zs)*dq3[B2] * REF_3 + 0.25*(double)(is)*dq1[B2] * REF_1;
			}
			else{
				ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = (receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + 0.25*(double)(zs)*dq3[NPR + 1] * REF_3 + 0.25*(double)(is)*dq1[NPR + 1] * REF_1) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
				psh[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] = (receive[nl[n_rec2]][(NPR + 3)*iz*isize*jsize + (NPR + 3)*ii*jsize + (NPR + 3)*ij + (1 + NPR)] + 0.25*(double)(zs)*dq3[NPR + 1] * REF_3 + 0.25*(double)(is)*dq1[NPR + 1] * REF_1) /
					gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2];
			}
			#endif
		}
	}
}

