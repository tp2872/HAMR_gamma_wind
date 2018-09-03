#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}
void pack_send1_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1)*(z2 - z1);
		 packsend1flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxsend1: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++) for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
				PLOOP send[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
				PLOOP send[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send2_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(z2 - z1);
		 packsend2flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxsend2: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				PLOOP send[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				PLOOP send[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send3_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, cudaEvent_t *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		 packsend3flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Error fluxsend3: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
				PLOOP send[nl[n]][NPR*(z - z1)*jsize*isize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] = factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
				PLOOP send[nl[n]][NPR*(z - z1)*jsize*isize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] += factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_flux_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1) / (1 + ref_2)*(z2 - z1) / (1 + ref_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1) / (1 + ref_2)*(z2 - z1) / (1 + ref_3);
		 packsendfluxaverage1 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxsendaverage1: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j += 1 + ref_2)for (z = z1; z < z2; z += (1 + ref_3)){
				PLOOP send[nl[n]][NPR*(i - i1) *zsize*jsize + NPR*(j - j1) / (1 + ref_2)*zsize + NPR*(z - z1) / (1 + ref_3) + k]
					= 0.25* factor*(F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]
					+ F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j += 1 + ref_2) for (z = z1; z < z2; z += (1 + ref_3)){
				PLOOP send[nl[n]][NPR*(i - i1) *zsize*jsize + NPR*(j - j1) / (1 + ref_2)*zsize + NPR*(z - z1) / (1 + ref_3) + k]
					+= 0.25* factor*(F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]
					+ F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_flux_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict F2[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + ref_1)*(z2 - z1) / (1 + ref_3)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + ref_1)*(z2 - z1) / (1 + ref_3);
		packsendfluxaverage2 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Error fluxsendaverage2: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + ref_1)for (z = z1; z < z2; z += 1 + ref_3){
				PLOOP send[nl[n]][NPR*(j - j1)*isize*zsize + NPR*(i - i1) / (1 + ref_1)*zsize + NPR*(z - z1) / (1 + ref_3) + k]
					= 0.25*factor*(F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k] +
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + ref_1)for (z = z1; z < z2; z += 1 + ref_3){
				PLOOP send[nl[n]][NPR*(j - j1)*isize*zsize + NPR*(i - i1) / (1 + ref_1)*zsize + NPR*(z - z1) / (1 + ref_3) + k]
					+= 0.25*factor*(F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k] +
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F2[nl[n]][index_3D(n, i + N1_GPU_offset[n] + ref_1, j + N2_GPU_offset[n], z + ref_3 + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_flux_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB_LOCAL], double(*restrict F3[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	int ref_1 = block[n][AMR_LEVEL1] - block[n_rec][AMR_LEVEL1];
	int ref_2 = block[n][AMR_LEVEL2] - block[n_rec][AMR_LEVEL2];
	int ref_3 = block[n][AMR_LEVEL3] - block[n_rec][AMR_LEVEL3];
	if (gpu == 1){
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1) / (1 + ref_1)*(j2 - j1) / (1 + ref_2)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1) / (1 + ref_1)*(j2 - j1) / (1 + ref_2);
		packsendfluxaverage3 << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundsend[0], factor, first_timestep, work_size, ref_1, ref_2, ref_3);
		 if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1 && block[n_rec][AMR_NODE] == block[n][AMR_NODE]){
			 cudaEventRecord(boundevent[0], commandQueueGPU[nl[n]]);
		 }
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Error fluxsendaverage3: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + ref_1)for (j = j1; j < j2; j += 1 + ref_2){
				PLOOP send[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1) / (1 + ref_1)*jsize + NPR*(j - j1) / (1 + ref_2) + k]
					= 0.25*factor*(F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + ref_1) for (j = j1; j < j2; j += 1 + ref_2){
				PLOOP send[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1) / (1 + ref_1)*jsize + NPR*(j - j1) / (1 + ref_2) + k]
					+= 0.25*factor*(F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F3[nl[n]][index_3D(n, i + ref_1 + N1_GPU_offset[n], j + ref_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void unpack_receive1_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		int nr_workgroups_bound = (int)ceil((double)((j2 - j1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (j2 - j1)*(z2 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive1flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, jsize, zsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxrec1: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= receive[nl[n_rec]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= (temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= (temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= receive[nl[n_rec]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp2[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp2[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= (receive[nl[n_rec]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - temp2[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= (receive[nl[n_rec]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - temp2[nl[n]][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]);
				}
			}
		}
	}
}

void unpack_receive2_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(z2 - z1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(z2 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive2flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, zsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxrec2: %d \n", status);

	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= receive[nl[n_rec]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor;
				}
			}
			else if (calc_corr == 5){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= receive[nl[n_rec]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp2[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp2[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= (receive[nl[n_rec]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - temp2[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= (receive[nl[n_rec]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - temp2[nl[n]][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]);
				}
			}
		}
	}
}

void unpack_receive3_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		int nr_workgroups_bound = (int)ceil((double)((i2 - i1)*(j2 - j1)) / ((double)(LOCAL_WORK_SIZE)));
		int work_size = (i2 - i1)*(j2 - j1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			if (block[n][AMR_NODE]==block[n_rec2][AMR_NODE]) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[0], 0);
		}
		 unpackreceive3flux << < nr_workgroups_bound, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (i1, i2, j1, j2, z1, z2, isize, jsize, Bufferp[0], Bufferboundreceive[0], Buffertemp1[0], Buffertemp2[0],
			calc_corr, nstep, block[n][AMR_NSTEP], timelevel, timelevel_rec, factor, work_size);
		 //cudaDeviceSynchronize();
		 status = cudaGetLastError();
		if (status != cudaSuccess) fprintf(stderr, "Error fluxrec3: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= receive[nl[n_rec]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= receive[nl[n_rec]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp2[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp2[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= factor*prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= (receive[nl[n_rec]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - temp2[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= (receive[nl[n_rec]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - temp2[nl[n]][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]);
				}
			}
		}
	}
}

