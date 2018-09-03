#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}
void pack_send1_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1)*(z2 - z1);
		 packsend1E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE]==block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsend1E: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (z = z1; z < z2; z++){
						k = 2;
						send[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (z = z1; z < z2; z++){
						k = 2;
						send[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send2_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(z2 - z1);
		 packsend2E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsend2E: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL]){
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						k = 1;
						send[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						k = 1;
						send[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send3_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		 packsend3E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsend3E: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
			for (z = z1; z < z2; z++){
				for (i = i1; i < i2; i++){
					for (j = j1; j < j2; j++){
						k = 1;
						send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 2;
						send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (z = z1; z < z2; z++){
				for (i = i1; i < i2; i++){
					for (j = j1; j < j2; j++){
						k = 1;
						send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 2;
						send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send_E_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]);
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1) / (1 + ref_2)*(z2 - z1) / (1 + ref_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1) / (1 + ref_2)*(z2 - z1) / (1 + ref_3);
		packsendEaverage1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendEaverage1: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j += 1 + ref_2){
					for (z = z1; z < z2; z += (1 + ref_3)){
						k = 2;
						send[nl[n]][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + ref_2)*zsize + 2 * (z - z1) / (1 + ref_3) + 0]
							= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
						k = 3;
						send[nl[n]][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + ref_2)*zsize + 2 * (z - z1) / (1 + ref_3) + 1]
							= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
					}
				}
			}
		}
		else{
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j += 1 + ref_2){
					for (z = z1; z < z2; z += (1 + ref_3)){
						k = 2;
						send[nl[n]][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + ref_2)*zsize + 2 * (z - z1) / (1 + ref_3) + 0]
							+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
						k = 3;
						send[nl[n]][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + ref_2)*zsize + 2 * (z - z1) / (1 + ref_3) + 1]
							+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
					}
				}
			}
		}
	}
}
void pack_send_E_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + ref_1)*(z2 - z1) / (1 + ref_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + ref_1)*(z2 - z1) / (1 + ref_3);
		packsendEaverage2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendEaverage2: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + ref_1) for (z = z1; z < z2; z += 1 + ref_3){
				k = 1;
				send[nl[n]][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + ref_1)*zsize + 2 * (z - z1) / (1 + ref_3) + 0]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 3;
				send[nl[n]][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + ref_1)*zsize + 2 * (z - z1) / (1 + ref_3) + 1]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + ref_1)for (z = z1; z < z2; z += 1 + ref_3){
				k = 1;
				send[nl[n]][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + ref_1)*zsize + 2 * (z - z1) / (1 + ref_3) + 0]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 3;
				send[nl[n]][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + ref_1)*zsize + 2 * (z - z1) / (1 + ref_3) + 1]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
			}
		}
	}
}

void pack_send_E_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + ref_1)*(j2 - j1) / (1 + ref_2)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + ref_1)*(j2 - j1) / (1 + ref_2);
		packsendEaverage3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendEaverage3: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + ref_1)for (j = j1; j < j2; j += 1 + ref_2){
				k = 1;
				send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + ref_1)*jsize + 2 * (j - j1) / (1 + ref_2) + 0]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 2;
				send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + ref_1)*jsize + 2 * (j - j1) / (1 + ref_2) + 1]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + ref_1)for (j = j1; j < j2; j += 1 + ref_2){
				k = 1;
				send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + ref_1)*jsize + 2 * (j - j1) / (1 + ref_2) + 0]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 2;
				send[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + ref_1)*jsize + 2 * (j - j1) / (1 + ref_2) + 1]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void unpack_receive1_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];

	if (gpu == 1){
		int j22 = j2 + 1;
		int z22 = z2 + D3;
		int nr_workgroups_bound = (int)ceil((double)((j22 - j1)*(z22 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j22 - j1)*(z22 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive1E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, d1, d2, e1, e2, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Unpack1e: %d \n", status);
	}
	else{
		int i, j, z;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += (temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += (temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= (temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= (temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = (receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]);
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = (receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 *PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += (receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]);
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += (receive[nl[n_rec]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - temp2[nl[n]][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]);
				}
			}
		}
	}
}

void unpack_receive2_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];

	if (gpu == 1){
		int i22 = i2 + 1;
		int z22 = z2 + D3;
		int nr_workgroups_bound = (int)ceil((double)((i22 - i1)*(z22 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i22 - i1)*(z22 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE] == block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive2E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, d1, d2, e1, e2, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error unpack_receive2_E %d \n", status);
	}
	else{
		int i, j, z;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]
						= receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (j = j1; j < j2; j++)for (i = i1 + d1; i < i2 + d2; i++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]
						= receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						+= (temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						-= (temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 4){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor;
					//prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(-receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor + prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]);
				}
			}
			if (calc_corr == 5){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]
						+= receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (j = j1; j < j2; j++)for (i = i1 + d1; i < i2 + d2; i++)for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]
						+= receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = (receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]);
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = (receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += (receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]);
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp1[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += (receive[nl[n_rec]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - temp2[nl[n]][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]);
				}
			}
		}
	}
}

void unpack_receive3_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	if (gpu == 1){
		int i22 = i2 + 1;
		int j22 = j2 + 1;
		int nr_workgroups_bound = (int)ceil((double)((i22 - i1)*(j22 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i22 - i1)*(j22 - j1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive3E << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, e1, e2, d1, d2, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error unpack_receive3_E %d \n", status);
	}
	else{
		int i, j, z;

		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]
						= receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]
						= receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]) / factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						+= (temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]) / factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						-= (temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]
						+= receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]
						+= receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = (receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]);
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = (receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += (receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]);
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += (receive[nl[n_rec]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - temp2[nl[n]][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]);
				}
			}
		}
	}
}

void pack_send_E1_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1);
		 packsendE1corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j, z, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE1corn %d \n", status);
	}
	else{
		int i, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				k = 1;
				send[nl[n]][(i - i1)]
					= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (i = i1; i < i2; i++){
				k = 1;
				send[nl[n]][(i - i1)]
					+= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E2_corn(int n, int n_rec, int i, int j1, int j2, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1);
		 packsendE2corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j1, j2, z, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE2corn %d \n", status);
	}
	else{
		int j, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++){
				k = 2;
				send[nl[n]][(j - j1)]
					= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (j = j1; j < j2; j++){
				k = 2;
				send[nl[n]][(j - j1)]
					+= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E3_corn(int n, int n_rec, int i, int j, int z1, int z2, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (z2 - z1);
		 packsendE3corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j, z1, z2, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE3corn %d \n", status);
	}
	else{
		int z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++){
				k = 3;
				send[nl[n]][z - z1]
					= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (z = z1; z < z2; z++){
				k = 3;
				send[nl[n]][z - z1]
					+= factor*E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E1_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + ref_1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + ref_1);
		packsendE1corncourse << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j, z, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE1corncourse %d \n", status);
	}
	else{
		int i, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i += (1 + ref_1)){
				k = 1;
				send[nl[n]][(i - i1) / (1 + ref_1)]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (i = i1; i < i2; i += (1 + ref_1)){
				k = 1;
				send[nl[n]][(i - i1) / (1 + ref_1)]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_E2_corn_course(int n, int n_rec, int i, int j1, int j2, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1) / (1 + ref_2)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1) / (1 + ref_2);
		packsendE2corncourse << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j1, j2, z, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_2);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE2corncourse %d \n", status);
	}
	else{
		int j, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j += (1 + ref_2)){
				k = 2;
				send[nl[n]][(j - j1) / (1 + ref_2)]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (j = j1; j < j2; j += (1 + ref_2)){
				k = 2;
				send[nl[n]][(j - j1) / (1 + ref_2)]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + ref_2, z + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_E3_corn_course(int n, int n_rec, int i, int j, int z1, int z2, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((z2 - z1) / (1 + ref_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (z2 - z1) / (1 + ref_3);
		packsendE3corncourse << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j, z1, z2, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);;
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error packsendE3corncourse %d \n", status);
	}
	else{
		int z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z += (1 + ref_3)){
				k = 3;
				send[nl[n]][(z - z1) / (1 + ref_3)]
					= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
			}
		}
		else{
			for (z = z1; z < z2; z += (1 + ref_3)){
				k = 3;
				send[nl[n]][(z - z1) / (1 + ref_3)]
					+= factor*0.5*(E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + ref_3)][k]);
			}
		}
	}
}

void unpack_receive_E1_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE])cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceiveE1corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j, z, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error receiveE1corn %d \n", status);
	}
	else{
		int i;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++){
					temp1[nl[n]][(i - i1)]
						= receive[nl[n_rec]][(i - i1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[nl[n]][(i - i1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[nl[n]][(i - i1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++){
					temp1[nl[n]][(i - i1)]
						+= receive[nl[n_rec]][(i - i1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++){
					temp2[nl[n]][(i - i1)] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++){
					temp2[nl[n]][(i - i1)] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++){
					temp1[nl[n]][(i - i1)] = (receive[nl[n_rec]][(i - i1)] - temp2[nl[n]][(i - i1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[nl[n]][(i - i1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[nl[n]][(i - i1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++){
					temp1[nl[n]][(i - i1)] += (receive[nl[n_rec]][(i - i1)] - temp2[nl[n]][(i - i1)]);
				}
			}
		}
	}
}

void unpack_receive_E2_corn(int n, int n_rec, int n_rec2, int i, int j1, int j2, int z, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]){
				cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
			}
		}
		 unpackreceiveE2corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j1, j2, z, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error receiveE2corn %d \n", status);
	}
	else{
		int j;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++){
					temp1[nl[n]][(j - j1)]
						= receive[nl[n_rec]][(j - j1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						+= (temp1[nl[n]][(j - j1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						-= (temp1[nl[n]][(j - j1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (j = j1; j < j2; j++){
					temp1[nl[n]][(j - j1)]
						+= receive[nl[n_rec]][(j - j1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++){
					temp2[nl[n]][(j - j1)] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++){
					temp2[nl[n]][(j - j1)] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++){
					temp1[nl[n]][(j - j1)] = (receive[nl[n_rec]][(j - j1)] - temp2[nl[n]][(j - j1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[nl[n]][(j - j1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[nl[n]][(j - j1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++){
					temp1[nl[n]][(j - j1)] += (receive[nl[n_rec]][(j - j1)] - temp2[nl[n]][(j - j1)]);
				}
			}
		}
	}
}

void unpack_receive_E3_corn(int n, int n_rec, int n_rec2, int i, int j, int z1, int z2, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];

	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (z2 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE])cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceiveE3corn << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i, j, z1, z2, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "unpack_receive_E3_corn: %d \n", status);
	}
	else{
		int z;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++){
					temp1[nl[n]][(z - z1)]
						= receive[nl[n_rec]][(z - z1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						+= (temp1[nl[n]][(z - z1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						-= (temp1[nl[n]][(z - z1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++){
					temp1[nl[n]][(z - z1)]
						+= receive[nl[n_rec]][(z - z1)] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++){
					temp2[nl[n]][(z - z1)] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++){
					temp2[nl[n]][(z - z1)] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++){
					temp1[nl[n]][(z - z1)] = (receive[nl[n_rec]][(z - z1)] - temp2[nl[n]][(z - z1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[nl[n]][(z - z1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[nl[n]][(z - z1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++){
					temp1[nl[n]][(z - z1)] += (receive[nl[n_rec]][(z - z1)] - temp2[nl[n]][(z - z1)]);
				}
			}
		}
	}
}
