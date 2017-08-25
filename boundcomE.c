#include "decs.h"

void pack_send1_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsend1E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend1E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend1E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend1E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend1E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend1E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend1E[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsend1E[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsend1E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend1E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend1E[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend1E[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (j2 - j1)*(z2 - z1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend1E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend1E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsend1E: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (z = z1; z < z2; z++){
						k = 2;
						send[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j++){
					for (z = z1; z < z2; z++){
						k = 2;
						send[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send2_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsend2E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend2E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend2E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend2E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend2E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend2E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend2E[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsend2E[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsend2E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend2E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend2E[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend2E[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1)*(z2 - z1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend2E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend2E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsend2E: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL]){
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						k = 1;
						send[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (j = j1; j < j2; j++){
				for (i = i1; i < i2; i++){
					for (z = z1; z < z2; z++){
						k = 1;
						send[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 3;
						send[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send3_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsend3E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend3E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend3E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend3E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend3E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend3E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend3E[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsend3E[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsend3E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend3E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend3E[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend3E[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1)*(j2 - j1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend3E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend3E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsend3E: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
			for (z = z1; z < z2; z++){
				for (i = i1; i < i2; i++){
					for (j = j1; j < j2; j++){
						k = 1;
						send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 2;
						send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
		else{
			for (z = z1; z < z2; z++){
				for (i = i1; i < i2; i++){
					for (j = j1; j < j2; j++){
						k = 1;
						send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
						k = 2;
						send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
					}
				}
			}
		}
	}
}

void pack_send_E_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	int first_timestep = (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]);
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int mode = 0;
		clSetKernelArg(kernel_packsendEaverage1[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendEaverage1[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendEaverage1[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendEaverage1[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendEaverage1[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendEaverage1[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendEaverage1[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsendEaverage1[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsendEaverage1[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendEaverage1[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendEaverage1[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendEaverage1[n], 11, sizeof(cl_int), &first_timestep);
		clSetKernelArg(kernel_packsendEaverage1[n], 12, sizeof(cl_int), &mode);

		global_work_size_bound[n][0] = (j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage1[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage1[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		clFlush(commandQueueGPU[n]);
		mode = 1;
		clSetKernelArg(kernel_packsendEaverage1[n], 12, sizeof(cl_int), &mode);
		status += clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage1[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		if (status != 0) printf("Error packsendaverage1: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j += 1 + REF_2){
					for (z = z1; z < z2; z += (1 + REF_3)){
						k = 2;
						send[n][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + REF_2)*zsize + 2 * (z - z1) / (1 + REF_3) + 0]
							= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
						k = 3;
						send[n][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + REF_2)*zsize + 2 * (z - z1) / (1 + REF_3) + 1]
							= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
					}
				}
			}
		}
		else{
			for (i = i1; i < i2; i++){
				for (j = j1; j < j2; j += 1 + REF_2){
					for (z = z1; z < z2; z += (1 + REF_3)){
						k = 2;
						send[n][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + REF_2)*zsize + 2 * (z - z1) / (1 + REF_3) + 0]
							+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
						k = 3;
						send[n][2 * (i - i1) *zsize*jsize + 2 * (j - j1) / (1 + REF_2)*zsize + 2 * (z - z1) / (1 + REF_3) + 1]
							+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
							E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
					}
				}
			}
		}

		double avg;
		//Here we average the intermediate of the electric fields at the fine-course boundary so that no artifacts are created by loss of causal connection
		/*for (i = i1; i < i2; i++)for (j = j1; j < j2; j += 1 + REF_2)for (z = z1; z < z2 - (1 + REF_3); z += (1 + REF_3)){
		k = 2;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + 2 * REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + 2 * REF_3)][k]);
		}
		for (i = i1; i < i2; i++)for (j = j1; j < j2 - 1 + REF_2; j += 1 + REF_2)for (z = z1; z < z2; z += (1 + REF_3)){
		k = 3;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + 2 * REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + 2 * REF_2, z + N3_GPU_offset[n] + REF_3)][k]);
		}

		for (i = i1; i < i2; i++) for (j = j1; j < j2; j += 1 + REF_2) for (z = z1; z < z2; z += (1 + REF_3)){
		k = 2;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k] = avg;

		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;

		k = 3;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n] + REF_3)][k] = avg;
		}*/
	}
}
void pack_send_E_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int mode = 0;
		clSetKernelArg(kernel_packsendEaverage2[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendEaverage2[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendEaverage2[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendEaverage2[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendEaverage2[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendEaverage2[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendEaverage2[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsendEaverage2[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsendEaverage2[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendEaverage2[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendEaverage2[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendEaverage2[n], 11, sizeof(cl_int), &first_timestep);
		clSetKernelArg(kernel_packsendEaverage2[n], 12, sizeof(cl_int), &mode);

		global_work_size_bound[n][0] = (i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage2[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage2[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		clFlush(commandQueueGPU[n]);
		mode = 1;
		clSetKernelArg(kernel_packsendEaverage2[n], 12, sizeof(cl_int), &mode);
		status += clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage2[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		if (status != 0) printf("Error packsendaverage2: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1) for (z = z1; z < z2; z += 1 + REF_3){
				k = 1;
				send[n][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + REF_1)*zsize + 2 * (z - z1) / (1 + REF_3) + 0]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 3;
				send[n][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + REF_1)*zsize + 2 * (z - z1) / (1 + REF_3) + 1]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1)for (z = z1; z < z2; z += 1 + REF_3){
				k = 1;
				send[n][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + REF_1)*zsize + 2 * (z - z1) / (1 + REF_3) + 0]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 3;
				send[n][2 * (j - j1)*isize*zsize + 2 * (i - i1) / (1 + REF_1)*zsize + 2 * (z - z1) / (1 + REF_3) + 1]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
			}
		}
		double avg;
		//Here we average the intermediate of the electric fields at the fine-course boundary so that no artifacts are created by loss of causal connection
		/*for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1)for (z = z1; z < z2 - (1 + REF_3); z += 1 + REF_3){
		k = 1;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + 2 * REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + 2 * REF_3)][k]);
		}
		for (j = j1; j < j2; j++)for (i = i1; i < i2 - (1 + REF_1); i += 1 + REF_1)for (z = z1; z < z2; z += 1 + REF_3){
		k = 3;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + 2 * REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + 2 * REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
		}
		for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1)for (z = z1; z < z2; z += 1 + REF_3){
		k = 1;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;

		k = 3;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		}*/
	}
}

void pack_send_E_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int mode = 0;
		clSetKernelArg(kernel_packsendEaverage3[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendEaverage3[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendEaverage3[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendEaverage3[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendEaverage3[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendEaverage3[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendEaverage3[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsendEaverage3[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsendEaverage3[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendEaverage3[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendEaverage3[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendEaverage3[n], 11, sizeof(cl_int), &first_timestep);
		clSetKernelArg(kernel_packsendEaverage3[n], 12, sizeof(cl_int), &mode);

		global_work_size_bound[n][0] = (j2 - j1) / (1 + REF_2)*(i2 - i1) / (1 + REF_1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage3[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage3[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		clFlush(commandQueueGPU[n]);
		mode = 1;
		clSetKernelArg(kernel_packsendEaverage3[n], 12, sizeof(cl_int), &mode);
		status += clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendEaverage3[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		if (status != 0) printf("Error packsendaverage2: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1)for (j = j1; j < j2; j += 1 + REF_2){
				k = 1;
				send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + REF_1)*jsize + 2 * (j - j1) / (1 + REF_2) + 0]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 2;
				send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + REF_1)*jsize + 2 * (j - j1) / (1 + REF_2) + 1]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1)for (j = j1; j < j2; j += 1 + REF_2){
				k = 1;
				send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + REF_1)*jsize + 2 * (j - j1) / (1 + REF_2) + 0]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
				k = 2;
				send[n][2 * (z - z1)*isize*jsize + 2 * (i - i1) / (1 + REF_1)*jsize + 2 * (j - j1) / (1 + REF_2) + 1]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
			}
		}
		double avg;
		//Here we average the intermediate of the electric fields at the fine-course boundary so that no artifacts are created by loss of causal connection
		/*for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1)for (j = j1; j < j2 - (1 + REF_2); j += 1 + REF_2){
		k = 1;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + 2 * REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + 2 * REF_2, z + N3_GPU_offset[n])][k]);
		}
		for (z = z1; z < z2; z++)for (i = i1; i < i2 - (1 + REF_1); i += 1 + REF_1)for (j = j1; j < j2; j += 1 + REF_2){
		k = 2;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + 2 * REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		= 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + 2 * REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
		}
		for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1)for (j = j1; j < j2; j += 1 + REF_2){
		k = 1;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;

		k = 2;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
		+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		}*/
	}
}

void unpack_receive1_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];

	if (gpu == 1){
		//if (n == AMR_coord_linear(1, 1, 4, 0) && n_rec == AMR_coord_linear(2, 4, 9, 0)) printf("test3: d1: %d d2: %d \n", d1, d2);
		//if (n == AMR_coord_linear(1, 2, 5, 0) && n_rec == AMR_coord_linear(1, 1, 5, 0)) printf("test4: d1: %d d2: %d \n", d1, d2);
		//if (n == 5)printf("errors %d %d %d \n", n_rec, j1 + d1, j2 + d2);
		int j22 = j2 + 1;
		int z22 = z2 + D3;
		clSetKernelArg(kernel_unpackreceive1E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive1E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive1E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive1E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive1E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive1E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive1E[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_unpackreceive1E[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_unpackreceive1E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive1E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive1E[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive1E[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive1E[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive1E[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive1E[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive1E[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive1E[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive1E[n], 17, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_unpackreceive1E[n], 18, sizeof(cl_int), &d1);
		clSetKernelArg(kernel_unpackreceive1E[n], 19, sizeof(cl_int), &d2);
		clSetKernelArg(kernel_unpackreceive1E[n], 20, sizeof(cl_int), &e1);
		clSetKernelArg(kernel_unpackreceive1E[n], 21, sizeof(cl_int), &e2);

		global_work_size_bound[n][0] = (j22 - j1)*(z22 - z1);
		//clWaitForEvents(1, &(boundevent[0]));
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive1E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive1E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Unpack1e: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += (temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += (temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= (temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= (temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] = (receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]);
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] = (receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] += (receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0] - temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 0]);
				}
				for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++)for (z = z1; z < z2; z++){
					temp1[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] += (receive[n_rec][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1] - temp2[n][2 * (i - i1)*zsize*jsize + 2 * (j - j1)*zsize + 2 * (z - z1) + 1]);
				}
			}
		}
	}
}

void unpack_receive2_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];

	if (gpu == 1){
		int i22 = i2 + 1;
		int z22 = z2 + D3;
		clSetKernelArg(kernel_unpackreceive2E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive2E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive2E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive2E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive2E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive2E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive2E[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_unpackreceive2E[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_unpackreceive2E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive2E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive2E[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive2E[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive2E[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive2E[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive2E[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive2E[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive2E[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive2E[n], 17, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_unpackreceive2E[n], 18, sizeof(cl_int), &d1);
		clSetKernelArg(kernel_unpackreceive2E[n], 19, sizeof(cl_int), &d2);
		clSetKernelArg(kernel_unpackreceive2E[n], 20, sizeof(cl_int), &e1);
		clSetKernelArg(kernel_unpackreceive2E[n], 21, sizeof(cl_int), &e2);
		global_work_size_bound[n][0] = (i22 - i1)*(z22 - z1);
		//clWaitForEvents(1, &(boundevent[0]));
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive2E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive2E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error unpack_receive2_E %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]
						= receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (j = j1; j < j2; j++)for (i = i1 + d1; i < i2 + d2; i++)for (z = z1; z < z2; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]
						= receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						+= (temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]) / factor;
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						-= (temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]) / factor;
				}
			}
			else if (calc_corr == 4){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] = receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor;
					//prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] = 0.5*(-receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor + prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]);
				}
			}
			if (calc_corr == 5){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]
						+= receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (j = j1; j < j2; j++)for (i = i1 + d1; i < i2 + d2; i++)for (z = z1; z < z2; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]
						+= receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] = (receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]);
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] = (receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++) for (i = i1; i < i2; i++) for (z = z1 + e1*D3; z < z2 + e2*D3; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] += (receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0] - temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 0]);
				}
				for (j = j1; j < j2; j++) for (i = i1 + d1; i < i2 + d2; i++) for (z = z1; z < z2; z++){
					temp1[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] += (receive[n_rec][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1] - temp2[n][2 * (j - j1)*zsize*isize + 2 * (i - i1)*zsize + 2 * (z - z1) + 1]);
				}
			}
		}
	}
}

void unpack_receive3_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr, int d1, int d2, int e1, int e2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	if (gpu == 1){
		int i22 = i2 + 1;
		int j22 = j2 + 1;
		clSetKernelArg(kernel_unpackreceive3E[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive3E[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive3E[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive3E[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive3E[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive3E[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive3E[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_unpackreceive3E[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_unpackreceive3E[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive3E[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive3E[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive3E[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive3E[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive3E[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive3E[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive3E[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive3E[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive3E[n], 17, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_unpackreceive3E[n], 18, sizeof(cl_int), &e1);
		clSetKernelArg(kernel_unpackreceive3E[n], 19, sizeof(cl_int), &e2);
		clSetKernelArg(kernel_unpackreceive3E[n], 20, sizeof(cl_int), &d1);
		clSetKernelArg(kernel_unpackreceive3E[n], 21, sizeof(cl_int), &d2);
		global_work_size_bound[n][0] = (j22 - j1)*(i22 - i1);
		//clWaitForEvents(1, &(boundevent[0]));
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive3E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive3E[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;

		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]
						= receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]
						= receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]) / factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						+= (temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]) / factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						-= (temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]
						+= receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]
						+= receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] = (receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]);
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] = (receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] / factor; //times dt_old/dt_new to add in future code
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1 + d1; j < j2 + d2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] += (receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0] - temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 0]);
				}
				for (z = z1; z < z2; z++)for (i = i1 + e1; i < i2 + e2; i++)for (j = j1; j < j2; j++){
					temp1[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] += (receive[n_rec][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1] - temp2[n][2 * (z - z1)*isize*jsize + 2 * (i - i1)*jsize + 2 * (j - j1) + 1]);
				}
			}
		}
	}
}

void pack_send_E1_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsendE1corn[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendE1corn[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendE1corn[n], 2, sizeof(cl_int), &j);
		clSetKernelArg(kernel_packsendE1corn[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_packsendE1corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE1corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE1corn[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE1corn[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE1corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE1corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE1corn %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++){
				k = 1;
				send[n][(i - i1)]
					= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (i = i1; i < i2; i++){
				k = 1;
				send[n][(i - i1)]
					+= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E2_corn(int n, int n_rec, int i, int j1, int j2, int z, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsendE2corn[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_packsendE2corn[n], 1, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendE2corn[n], 2, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendE2corn[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_packsendE2corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE2corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE2corn[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE2corn[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (j2 - j1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE2corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE2corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE2corn %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int j, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++){
				k = 2;
				send[n][(j - j1)]
					= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (j = j1; j < j2; j++){
				k = 2;
				send[n][(j - j1)]
					+= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E3_corn(int n, int n_rec, int i, int j, int z1, int z2, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsendE3corn[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_packsendE3corn[n], 1, sizeof(cl_int), &j);
		clSetKernelArg(kernel_packsendE3corn[n], 2, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendE3corn[n], 3, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendE3corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE3corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE3corn[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE3corn[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (z2 - z1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE3corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE3corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE2corn %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++){
				k = 3;
				send[n][z - z1]
					= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (z = z1; z < z2; z++){
				k = 3;
				send[n][z - z1]
					+= factor*E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_E1_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsendE1corncourse[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendE1corncourse[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendE1corncourse[n], 2, sizeof(cl_int), &j);
		clSetKernelArg(kernel_packsendE1corncourse[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_packsendE1corncourse[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE1corncourse[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE1corncourse[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE1corncourse[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1) / (1 + REF_1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE1corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE1corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE1corncourse %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i += (1 + REF_1)){
				k = 1;
				send[n][(i - i1) / (1 + REF_1)]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (i = i1; i < i2; i += (1 + REF_1)){
				k = 1;
				send[n][(i - i1) / (1 + REF_1)]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
		double avg;
		for (i = i1; i < i2; i += (1 + REF_1)){
			k = 1;
			//avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
			//	+ E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			//E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
			//E[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
		}
	}
}

void pack_send_E2_corn_course(int n, int n_rec, int i, int j1, int j2, int z, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];

	if (gpu == 1){
		clSetKernelArg(kernel_packsendE2corncourse[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_packsendE2corncourse[n], 1, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendE2corncourse[n], 2, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendE2corncourse[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_packsendE2corncourse[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE2corncourse[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE2corncourse[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE2corncourse[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (j2 - j1) / (1 + REF_2);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE2corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE2corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE2corncourse %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int j, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j += (1 + REF_2)){
				k = 2;
				send[n][(j - j1) / (1 + REF_2)]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (j = j1; j < j2; j += (1 + REF_2)){
				k = 2;
				send[n][(j - j1) / (1 + REF_2)]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
			}
		}
		double avg;
		for (j = j1; j < j2; j += (1 + REF_2)){
			k = 2;
			//avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
			//	+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k]);
			//E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
			//E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + REF_2, z + N3_GPU_offset[n])][k] = avg;
		}
	}
}

void pack_send_E3_corn_course(int n, int n_rec, int i, int j, int z1, int z2, double *send[NB], double(*restrict E[NB])[NDIM], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		//if (n == 21)printf("errors %d %d %d \n", n_rec,i,j);

		clSetKernelArg(kernel_packsendE3corncourse[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_packsendE3corncourse[n], 1, sizeof(cl_int), &j);
		clSetKernelArg(kernel_packsendE3corncourse[n], 2, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendE3corncourse[n], 3, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendE3corncourse[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendE3corncourse[n], 5, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendE3corncourse[n], 6, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendE3corncourse[n], 7, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (z2 - z1) / (1 + REF_3);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE3corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendE3corncourse[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error packsendE3corncourse %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z += (1 + REF_3)){
				k = 3;
				send[n][(z - z1) / (1 + REF_3)]
					= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
			}
		}
		else{
			for (z = z1; z < z2; z += (1 + REF_3)){
				k = 3;
				send[n][(z - z1) / (1 + REF_3)]
					+= factor*0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
			}
		}
		double avg;
		for (z = z1; z < z2; z += (1 + REF_3)){
			k = 3;
			//avg = 0.5*(E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
			//	+ E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k]);
			//E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = avg;
			//E[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + REF_3)][k] = avg;
		}
	}
}

void unpack_receive_E1_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 2, sizeof(cl_int), &j);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 6, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 7, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 8, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 9, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 10, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 11, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 12, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceiveE1corn[n], 13, sizeof(cl_double), &factor);
		global_work_size_bound[n][0] = i2 - i1;
		//clWaitForEvents(1, &boundevent[0]);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE1corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE1corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error receiveE1corn %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++){
					temp1[n][(i - i1)]
						= receive[n_rec][(i - i1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						+= (temp1[n][(i - i1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1]
						-= (temp1[n][(i - i1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++){
					temp1[n][(i - i1)]
						+= receive[n_rec][(i - i1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++){
					temp2[n][(i - i1)] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++){
					temp2[n][(i - i1)] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++){
					temp1[n][(i - i1)] = (receive[n_rec][(i - i1)] - temp2[n][(i - i1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] += temp1[n][(i - i1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][1] -= temp1[n][(i - i1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++){
					temp1[n][(i - i1)] += (receive[n_rec][(i - i1)] - temp2[n][(i - i1)]);
				}
			}
		}
	}
}

void unpack_receive_E2_corn(int n, int n_rec, int n_rec2, int i, int j1, int j2, int z, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 1, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 2, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 3, sizeof(cl_int), &z);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 6, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 7, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 8, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 9, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 10, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 11, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 12, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceiveE2corn[n], 13, sizeof(cl_double), &factor);
		global_work_size_bound[n][0] = j2 - j1;
		//clWaitForEvents(1, &boundevent[0]);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE2corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE2corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error receiveE2corn %d \n", status);

		clFlush(commandQueueGPU[n]);
	}
	else{
		int j, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++){
					temp1[n][(j - j1)]
						= receive[n_rec][(j - j1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						+= (temp1[n][(j - j1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2]
						-= (temp1[n][(j - j1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (j = j1; j < j2; j++){
					temp1[n][(j - j1)]
						+= receive[n_rec][(j - j1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++){
					temp2[n][(j - j1)] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++){
					temp2[n][(j - j1)] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++){
					temp1[n][(j - j1)] = (receive[n_rec][(j - j1)] - temp2[n][(j - j1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] += temp1[n][(j - j1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][2] -= temp1[n][(j - j1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++){
					temp1[n][(j - j1)] += (receive[n_rec][(j - j1)] - temp2[n][(j - j1)]);
				}
			}
		}
	}
}

void unpack_receive_E3_corn(int n, int n_rec, int n_rec2, int i, int j, int z1, int z2, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent, int calc_corr){
	int timelevel = block[n][AMR_TIMELEVEL];
	int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
	double factor = dt*(double)block[n][AMR_TIMELEVEL];

	if (gpu == 1){
		//if (n == 0)printf("errors %d %d %d \n", n_rec,i,j);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 0, sizeof(cl_int), &i);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 1, sizeof(cl_int), &j);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 2, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 3, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 4, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 5, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 6, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 7, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 8, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 9, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 10, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 11, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 12, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceiveE3corn[n], 13, sizeof(cl_double), &factor);
		global_work_size_bound[n][0] = z2 - z1;
		//clWaitForEvents(1, &boundevent[0]);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE3corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent != NULL), boundevent, NULL);
			if (boundevent != NULL) clReleaseEvent(boundevent[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceiveE3corn[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("unpack_receive_E3_corn: %d \n", status);


		clFlush(commandQueueGPU[n]);
	}
	else{
		int z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++){
					temp1[n][(z - z1)]
						= receive[n_rec][(z - z1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						+= (temp1[n][(z - z1)]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3]
						-= (temp1[n][(z - z1)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++){
					temp1[n][(z - z1)]
						+= receive[n_rec][(z - z1)] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++){
					temp2[n][(z - z1)] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++){
					temp2[n][(z - z1)] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3];
				}
			}
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++){
					temp1[n][(z - z1)] = (receive[n_rec][(z - z1)] - temp2[n][(z - z1)]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] += temp1[n][(z - z1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][3] -= temp1[n][(z - z1)] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++){
					temp1[n][(z - z1)] += (receive[n_rec][(z - z1)] - temp2[n][(z - z1)]);
				}
			}
		}
	}
}

/*Send boundaries of Ees between compute nodes through MPI*/
void E_send1(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send1_E(n, block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] + N2G, 0, N3_GPU[n] + N3G, (N2_GPU[n] + N2G), (N3_GPU[n] + N3G), send2_E, E, &(Bufferp[n]), &(Buffersend2E[n]),
				&(boundevent[n][220]));
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][220]);
					clReleaseEvent(boundevent[n][220]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend2E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G)*sizeof(double), send2_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send2_E[n][0], 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (20 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_E[n][0], 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((40 * NB) + block[n][AMR_NBR2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][240]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive4_5E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((40 * NB) + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][245]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive4_6E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((40 * NB) + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][246]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive4_7E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((40 * NB) + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][247]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive4_8E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((40 * NB) + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][248]);
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + D1, 0, N2_GPU[n] + N2G, 0, N3_GPU[n] + N3G,
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send2_E, E,
					&(Bufferp[n]), &(Buffersend2E[n]), &(boundevent[n][220]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][220]);
						clReleaseEvent(boundevent[n][220]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend2E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send2_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send2_E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((20 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[38]);
					MPI_Request_free(&req[38]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send1_E(n, block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n] + N2G, 0, N3_GPU[n] + N3G, (N2_GPU[n] + N2G), (N3_GPU[n] + N3G), send4_E, E, &(Bufferp[n]), &(Buffersend4E[n]),
				&(boundevent[n][240]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][240]);
					clReleaseEvent(boundevent[n][240]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend4E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G)*sizeof(double), send4_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send4_E[n][0], 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((40 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_E[n][0], 2 * (N3_GPU[n] + N3G)*(N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((20 * NB) + block[n][AMR_NBR4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][220]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_1E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((20 * NB) + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][221]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_2E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((20 * NB) + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][222]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_3E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((20 * NB) + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][223]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_4E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((20 * NB) + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][224]);
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, 1, 0, N2_GPU[n] + N2G, 0, N3_GPU[n] + N3G,
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send4_E, E,
					&(Bufferp[n]), &(Buffersend4E[n]), &(boundevent[n][240]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][240]);
						clReleaseEvent(boundevent[n][240]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend4E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send4_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send4_E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((40 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[39]);
					MPI_Request_free(&req[39]);
				}
			}
		}
	}
#endif
}

void E_send2(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send2_E(n, block[n][AMR_NBR3], 0, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] + N3G, (N1_GPU[n] + N1G), (N3_GPU[n] + N3G), send3_E, E, &(Bufferp[n]), &(Buffersend3E[n]),
				&(boundevent[n][230]));
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][230]);
					clReleaseEvent(boundevent[n][230]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend3E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N1G)*(N1_GPU[n] + N1G)*sizeof(double), send3_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send3_E[n][0], 2 * (N3_GPU[n] + N3G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((30 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_E[n][0], 2 * (N3_GPU[n] + N3G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((10 * NB) + block[n][AMR_NBR3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][210]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive1_3E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((10 * NB) + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][213]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive1_4E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((10 * NB) + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][214]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive1_7E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((10 * NB) + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][217]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive1_8E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((10 * NB) + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][218]);
		}
		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], 0, N1_GPU[n] + N1G, N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] + N3G,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send3_E, E,
					&(Bufferp[n]), &(Buffersend3E[n]), &(boundevent[n][230]));
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][230]);
						clReleaseEvent(boundevent[n][230]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend3E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send3_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send3_E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((30 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[40]);
					MPI_Request_free(&req[40]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send2_E(n, block[n][AMR_NBR1], 0, N1_GPU[n] + N1G, 0, 1, 0, N3_GPU[n] + N3G, (N1_GPU[n] + N1G), (N3_GPU[n] + N3G), send1_E, E, &(Bufferp[n]), &(Buffersend1E[n]),
				&(boundevent[n][210]));
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][210]);
					clReleaseEvent(boundevent[n][210]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend1E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G)*(N1_GPU[n] + N1G)*sizeof(double), send1_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send1_E[n][0], 2 * (N3_GPU[n] + N3G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((10 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_E[n][0], 2 * (N3_GPU[n] + N3G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((30 * NB) + block[n][AMR_NBR1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][230]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_1E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((30 * NB) + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][231]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_2E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((30 * NB) + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][232]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_5E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((30 * NB) + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][235]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_6E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((30 * NB) + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][236]);
		}
		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], 0, N1_GPU[n] + N1G, 0, 1, 0, N3_GPU[n] + N3G,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send1_E, E,
					&(Bufferp[n]), &(Buffersend1E[n]), &(boundevent[n][210]));
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][210]);
						clReleaseEvent(boundevent[n][210]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend1E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send1_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send1_E[n][0], 2 * (N3_GPU[n] + N3G) / (1 + REF_3)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((10 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[41]);
					MPI_Request_free(&req[41]);
				}
			}
		}
	}
#endif
}

void E_send3(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send3_E(n, block[n][AMR_NBR5], 0, N1_GPU[n] + N1G, 0, N2_GPU[n] + N2G, N3_GPU[n], N3_GPU[n] + D3, (N1_GPU[n] + N1G), (N2_GPU[n] + N2G), send5_E, E, &(Bufferp[n]), &(Buffersend5E[n]),
				&(boundevent[n][250]));
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][250]);
					clReleaseEvent(boundevent[n][250]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend5E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G)*sizeof(double), send5_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send5_E[n][0], 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((50 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_E[n][0], 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((60 * NB) + block[n][AMR_NBR5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][260]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive6_2E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((60 * NB) + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][262]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive6_4E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((60 * NB) + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][264]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive6_6E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((60 * NB) + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][266]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive6_8E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((60 * NB) + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][268]);
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], 0, N1_GPU[n] + N1G, 0, N2_GPU[n] + N2G, N3_GPU[n], N3_GPU[n] + D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send5_E, E,
					&(Bufferp[n]), &(Buffersend5E[n]), &(boundevent[n][250]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][250]);
						clReleaseEvent(boundevent[n][250]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend5E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send5_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send5_E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((50 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[42]);
					MPI_Request_free(&req[42]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send3_E(n, block[n][AMR_NBR6], 0, N1_GPU[n] + N1G, 0, N2_GPU[n] + N2G, 0, D3, (N1_GPU[n] + N1G), (N2_GPU[n] + N2G), send6_E, E, &(Bufferp[n]), &(Buffersend6E[n]),
				&(boundevent[n][260]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][260]);
					clReleaseEvent(boundevent[n][260]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend6E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G)*sizeof(double), send6_E[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send6_E[n][0], 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((60 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_E[n][0], 2 * (N2_GPU[n] + N2G)*(N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((50 * NB) + block[n][AMR_NBR6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][250]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive5_1E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((50 * NB) + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][251]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive5_3E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((50 * NB) + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][253]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive5_5E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((50 * NB) + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][255]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive5_7E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((50 * NB) + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][257]);
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], 0, N1_GPU[n] + N1G, 0, N2_GPU[n] + N2G, 0, D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send6_E, E,
					&(Bufferp[n]), &(Buffersend6E[n]), &(boundevent[n][260]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][260]);
						clReleaseEvent(boundevent[n][260]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend6E[n], CL_TRUE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send6_E[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send6_E[n][0], 2 * (N2_GPU[n] + N2G) / (1 + REF_2)*(N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((60 * NB) + n) % MPI_TAG_MAX, mpi_cartcomm, &req[43]);
					MPI_Request_free(&req[43]);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E_rec1(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][220], &Statbound[n][220]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) *(N2_GPU[n] + N2G)*sizeof(double), receive2_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_E(n, n, block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n] + N2G, N3_GPU[n] + N3G, receive2_E, receive2_E1, NULL, E,
					&(Bufferp[n]), &(Bufferrec2E[n]), &(Bufferrec2E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR4] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR4] && block[n][AMR_CORN3D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) + 3 * (block[n][AMR_CORN3D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR4] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR4] && block[n][AMR_CORN7D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) + 3 * (block[n][AMR_CORN7D] == -100));
			}
			else{
				unpack_receive1_E(n, block[n][AMR_NBR4], block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n] + N2G, N3_GPU[n] + N3G, send2_E, receive2_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend2E[block[n][AMR_NBR4]]), &(Bufferrec2E1[n]), NULL, &(boundevent[block[n][AMR_NBR4]][220]), calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR4] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR4] && block[n][AMR_CORN3D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) + 3 * (block[n][AMR_CORN3D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR4] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR4] && block[n][AMR_CORN7D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) + 3 * (block[n][AMR_CORN7D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][221], &Statbound[n][221]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_1E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive2_1E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive2_1E, receive2_1E1, receive2_1E2, E,
					&(Bufferp[n]), &(Bufferrec2_1E[n]), &(Bufferrec2_1E1[n]), &(Bufferrec2_1E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD5], block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send2_E, receive2_1E1, receive2_1E2, E,
					&(Bufferp[n]), &(Buffersend2E[block[block[n][AMR_NBR4]][AMR_CHILD5]]), &(Bufferrec2_1E1[n]), &(Bufferrec2_1E2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD5]][220]), calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][222], &Statbound[n][222]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_2E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive2_2E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive2_2E, receive2_2E1, receive2_2E2, E,
						&(Bufferp[n]), &(Bufferrec2_2E[n]), &(Bufferrec2_2E1[n]), &(Bufferrec2_2E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN7D_1] == -100)));
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD6], block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send2_E, receive2_2E1, receive2_2E2, E,
						&(Bufferp[n]), &(Buffersend2E[block[block[n][AMR_NBR4]][AMR_CHILD6]]), &(Bufferrec2_2E1[n]), &(Bufferrec2_2E2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD6]][220]), calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN7D_1] == -100)));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][223], &Statbound[n][223]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_3E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive2_3E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive2_3E, receive2_3E1, receive2_3E2, E,
						&(Bufferp[n]), &(Bufferrec2_3E[n]), &(Bufferrec2_3E1[n]), &(Bufferrec2_3E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN3D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD7], block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send2_E, receive2_3E1, receive2_3E2, E,
						&(Bufferp[n]), &(Buffersend2E[block[block[n][AMR_NBR4]][AMR_CHILD7]]), &(Bufferrec2_3E1[n]), &(Bufferrec2_3E2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD7]][220]), calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN3D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_3 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][224], &Statbound[n][224]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_4E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive2_4E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive2_4E, receive2_4E1, receive2_4E2, E,
						&(Bufferp[n]), &(Bufferrec2_4E[n]), &(Bufferrec2_4E1[n]), &(Bufferrec2_4E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN3D_2] == -100)),
						block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN7D_2] == -100)));
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD8], block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send2_E, receive2_4E1, receive2_4E2, E,
						&(Bufferp[n]), &(Buffersend2E[block[block[n][AMR_NBR4]][AMR_CHILD8]]), &(Bufferrec2_4E1[n]), &(Bufferrec2_4E2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD8]][220]), calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN3D_2] == -100)),
						block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN7D_2] == -100)));
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][240], &Statbound[n][240]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) *(N2_GPU[n] + N2G)*sizeof(double), receive4_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_E(n, n, block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n] + N2G, N3_GPU[n] + N3G, receive4_E, receive4_E1, NULL, E, &(Bufferp[n]), &(Bufferrec4E[n]), &(Bufferrec4E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR2] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) - 3 * (block[n][AMR_CORN1D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR2] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR2] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) - 3 * (block[n][AMR_CORN5D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR2] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
			else{
				unpack_receive1_E(n, block[n][AMR_NBR2], block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n] + N2G, N3_GPU[n] + N3G, send4_E, receive4_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend4E[block[n][AMR_NBR2]]), &(Bufferrec4E1[n]), NULL, &(boundevent[block[n][AMR_NBR2]][240]), calc_corr,
					3 - (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR2] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) - 3 * (block[n][AMR_CORN1D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR2] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR2] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) - 3 * (block[n][AMR_CORN5D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR2] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][245], &Statbound[n][245]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_5E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive4_5E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive4_5E, receive4_5E1, receive4_5E2, E,
					&(Bufferp[n]), &(Bufferrec4_5E[n]), &(Bufferrec4_5E1[n]), &(Bufferrec4_5E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN1D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN5D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD1], block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send4_E, receive4_5E1, receive4_5E2, E,
					&(Bufferp[n]), &(Buffersend4E[block[block[n][AMR_NBR2]][AMR_CHILD1]]), &(Bufferrec4_5E1[n]), &(Bufferrec4_5E2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD1]][240]), calc_corr,
					1 - ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN1D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN5D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][246], &Statbound[n][246]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_6E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive4_6E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive4_6E, receive4_6E1, receive4_6E2, E,
						&(Bufferp[n]), &(Bufferrec4_6E[n]), &(Bufferrec4_6E1[n]), &(Bufferrec4_6E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN1D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN6D_1] == -100)));
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD2], block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send4_E, receive4_6E1, receive4_6E2, E,
						&(Bufferp[n]), &(Buffersend4E[block[block[n][AMR_NBR2]][AMR_CHILD2]]), &(Bufferrec4_6E1[n]), &(Bufferrec4_6E2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD2]][240]), calc_corr,
						1 - ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN1D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN6D_1] == -100)));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][247], &Statbound[n][247]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_7E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive4_7E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive4_7E, receive4_7E1, receive4_7E2, E,
						&(Bufferp[n]), &(Bufferrec4_7E[n]), &(Bufferrec4_7E1[n]), &(Bufferrec4_7E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN5D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD3], block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send4_E, receive4_7E1, receive4_7E2, E,
						&(Bufferp[n]), &(Buffersend4E[block[block[n][AMR_NBR2]][AMR_CHILD3]]), &(Bufferrec4_7E1[n]), &(Bufferrec4_7E2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD3]][240]), calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN5D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
			if (REF_3 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][248], &Statbound[n][248]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_8E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive4_8E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), receive4_8E, receive4_8E1, receive4_8E2, E,
						&(Bufferp[n]), &(Bufferrec4_8E[n]), &(Bufferrec4_8E1[n]), &(Bufferrec4_8E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN6D_2] == -100)));
				}
				else{
					unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD4], block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N2_GPU[n] + N2G) / (1 + REF_2), (N3_GPU[n] + N3G) / (1 + REF_3), send4_E, receive4_8E1, receive4_8E2, E,
						&(Bufferp[n]), &(Buffersend4E[block[block[n][AMR_NBR2]][AMR_CHILD4]]), &(Bufferrec4_8E1[n]), &(Bufferrec4_8E2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD4]][240]), calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN6D_2] == -100)));
				}
			}
		}
	}
#endif
}
void E_rec2(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][230], &Statbound[n][230]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N1_GPU[n] + N1G) *(N3_GPU[n] + N3G)*sizeof(double), receive3_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_E(n, n, block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n] + N1G, N3_GPU[n] + N3G, receive3_E, receive3_E1, NULL, E, &(Bufferp[n]), &(Bufferrec3E[n]), &(Bufferrec3E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR1] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR1] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) + 3 * (block[n][AMR_CORN1D] == -100),
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR1] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR1] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) + 3 * (block[n][AMR_CORN9D] == -100));
			}
			else{
				unpack_receive2_E(n, block[n][AMR_NBR1], block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n] + N1G, N3_GPU[n] + N3G, send3_E, receive3_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend3E[block[n][AMR_NBR1]]), &(Bufferrec3E1[n]), NULL, &(boundevent[block[n][AMR_NBR1]][230]), calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR1] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR1] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) + 3 * (block[n][AMR_CORN1D] == -100),
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR1] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR1] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) + 3 * (block[n][AMR_CORN9D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][231], &Statbound[n][231]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_1E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive3_1E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), 0, 1, 0, (N3_GPU[n]) / (1 + REF_3),
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive3_1E, receive3_1E1, receive3_1E2, E,
					&(Bufferp[n]), &(Bufferrec3_1E[n]), &(Bufferrec3_1E1[n]), &(Bufferrec3_1E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD3], block[block[n][AMR_NBR1]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), 0, 1, 0, (N3_GPU[n]) / (1 + REF_3),
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send3_E, receive3_1E1, receive3_1E2, E,
					&(Bufferp[n]), &(Buffersend3E[block[block[n][AMR_NBR1]][AMR_CHILD3]]), &(Bufferrec3_1E1[n]), &(Bufferrec3_1E2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD3]][230]), calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][232], &Statbound[n][232]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_2E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive3_2E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive3_2E, receive3_2E1, receive3_2E2, E,
						&(Bufferp[n]), &(Bufferrec3_2E[n]), &(Bufferrec3_2E1[n]), &(Bufferrec3_2E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN9D_1] == -100)));
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD4], block[block[n][AMR_NBR1]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send3_E, receive3_2E1, receive3_2E2, E,
						&(Bufferp[n]), &(Buffersend3E[block[block[n][AMR_NBR1]][AMR_CHILD4]]), &(Bufferrec3_2E1[n]), &(Bufferrec3_2E2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD4]][230]), calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN9D_1] == -100)));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][235], &Statbound[n][235]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_5E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive3_5E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, (N3_GPU[n]) / (1 + REF_3),
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive3_5E, receive3_5E1, receive3_5E2, E,
						&(Bufferp[n]), &(Bufferrec3_5E[n]), &(Bufferrec3_5E1[n]), &(Bufferrec3_5E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN1D_1] == -100)),
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD7], block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, (N3_GPU[n]) / (1 + REF_3),
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send3_E, receive3_5E1, receive3_5E2, E,
						&(Bufferp[n]), &(Buffersend3E[block[block[n][AMR_NBR1]][AMR_CHILD7]]), &(Bufferrec3_5E1[n]), &(Bufferrec3_5E2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD7]][230]), calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN1D_1] == -100)),
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][236], &Statbound[n][236]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_6E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive3_6E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive3_6E, receive3_6E1, receive3_6E2, E,
						&(Bufferp[n]), &(Bufferrec3_6E[n]), &(Bufferrec3_6E1[n]), &(Bufferrec3_6E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN1D_2] == -100)),
						block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN9D_2] == -100)));
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD8], block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send3_E, receive3_6E1, receive3_6E2, E,
						&(Bufferp[n]), &(Buffersend3E[block[block[n][AMR_NBR1]][AMR_CHILD8]]), &(Bufferrec3_6E1[n]), &(Bufferrec3_6E2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD8]][230]), calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN1D_2] == -100)),
						block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN9D_2] == -100)));
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][210], &Statbound[n][210]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N1_GPU[n] + N1G) *(N3_GPU[n] + N3G)*sizeof(double), receive1_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_E(n, n, block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n] + N1G, N3_GPU[n] + N3G, receive1_E, receive1_E1, NULL, E, &(Bufferp[n]), &(Bufferrec1E[n]), &(Bufferrec1E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR3] && block[n][AMR_CORN3D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) - 3 * (block[n][AMR_CORN3D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR3] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR3] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) - 3 * (block[n][AMR_CORN11D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR3] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100));
			}
			else{
				unpack_receive2_E(n, block[n][AMR_NBR3], block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n] + N1G, N3_GPU[n] + N3G, send1_E, receive1_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend1E[block[n][AMR_NBR3]]), &(Bufferrec1E1[n]), NULL, &(boundevent[block[n][AMR_NBR3]][210]), calc_corr,
					3 - (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR3] && block[n][AMR_CORN3D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) - 3 * (block[n][AMR_CORN3D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR3] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR3] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) - 3 * (block[n][AMR_CORN11D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR3] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][213], &Statbound[n][213]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_3E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive1_3E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, (N3_GPU[n]) / (1 + REF_3),
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive1_3E, receive1_3E1, receive1_3E2, E,
					&(Bufferp[n]), &(Bufferrec1_3E[n]), &(Bufferrec1_3E1[n]), &(Bufferrec1_3E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN3D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN11D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD1], block[block[n][AMR_NBR3]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, (N3_GPU[n]) / (1 + REF_3),
					(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send1_E, receive1_3E1, receive1_3E2, E,
					&(Bufferp[n]), &(Buffersend1E[block[block[n][AMR_NBR3]][AMR_CHILD1]]), &(Bufferrec1_3E1[n]), &(Bufferrec1_3E2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD1]][210]), calc_corr,
					1 - ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN3D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN11D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][214], &Statbound[n][214]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_4E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive1_4E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive1_4E, receive1_4E1, receive1_4E2, E,
						&(Bufferp[n]), &(Bufferrec1_4E[n]), &(Bufferrec1_4E1[n]), &(Bufferrec1_4E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN3D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN10D_1] == -100)));
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD2], block[block[n][AMR_NBR3]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send1_E, receive1_4E1, receive1_4E2, E,
						&(Bufferp[n]), &(Buffersend1E[block[block[n][AMR_NBR3]][AMR_CHILD2]]), &(Bufferrec1_4E1[n]), &(Bufferrec1_4E2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD2]][210]), calc_corr,
						1 - ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN3D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN10D_1] == -100)));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][217], &Statbound[n][217]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_7E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive1_7E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, (N3_GPU[n]) / (1 + REF_3),
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive1_7E, receive1_7E1, receive1_7E2, E,
						&(Bufferp[n]), &(Bufferrec1_7E[n]), &(Bufferrec1_7E1[n]), &(Bufferrec1_7E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN11D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD5], block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, (N3_GPU[n]) / (1 + REF_3),
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send1_E, receive1_7E1, receive1_7E2, E,
						&(Bufferp[n]), &(Buffersend1E[block[block[n][AMR_NBR3]][AMR_CHILD5]]), &(Bufferrec1_7E1[n]), &(Bufferrec1_7E2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD5]][210]), calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN11D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][218], &Statbound[n][218]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_8E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N3_GPU[n] + N3G) / (1 + REF_3) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive1_8E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), receive1_8E, receive1_8E1, receive1_8E2, E,
						&(Bufferp[n]), &(Bufferrec1_8E[n]), &(Bufferrec1_8E1[n]), &(Bufferrec1_8E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN10D_2] == -100)));
				}
				else{
					unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD6], block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						(N1_GPU[n] + N1G) / (1 + REF_1), (N3_GPU[n] + N3G) / (1 + REF_3), send1_E, receive1_8E1, receive1_8E2, E,
						&(Bufferp[n]), &(Buffersend1E[block[block[n][AMR_NBR3]][AMR_CHILD6]]), &(Bufferrec1_8E1[n]), &(Bufferrec1_8E2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD6]][210]), calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN10D_2] == -100)));
				}
			}
		}
	}
#endif
}
void E_rec3(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][250], &Statbound[n][250]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N1_GPU[n] + N1G) *(N2_GPU[n] + N2G)*sizeof(double), receive5_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_E(n, n, block[n][AMR_NBR6], 0, N1_GPU[n], 0, N2_GPU[n], 0, D3,
					N1_GPU[n] + N1G, N2_GPU[n] + N2G, receive5_E, receive5_E1, NULL, E, &(Bufferp[n]), &(Bufferrec5E[n]), &(Bufferrec5E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR6] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR6] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) + 3 * (block[n][AMR_CORN11D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR6] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR6] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) + 3 * (block[n][AMR_CORN5D] == -100));
			}
			else{
				unpack_receive3_E(n, block[n][AMR_NBR6], block[n][AMR_NBR6], 0, N1_GPU[n], 0, N2_GPU[n], 0, D3,
					N1_GPU[n] + N1G, N2_GPU[n] + N2G, send5_E, receive5_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend5E[block[n][AMR_NBR6]]), &(Bufferrec5E1[n]), NULL, &(boundevent[block[n][AMR_NBR6]][250]), calc_corr,
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR6] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR6] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) + 3 * (block[n][AMR_CORN11D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR6] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR6] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) + 3 * (block[n][AMR_CORN5D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][251], &Statbound[n][251]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_1E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive5_1E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), 0, (N2_GPU[n]) / (1 + REF_2), 0, D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive5_1E, receive5_1E1, receive5_1E2, E,
					&(Bufferp[n]), &(Bufferrec5_1E[n]), &(Bufferrec5_1E1[n]), &(Bufferrec5_1E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD2], block[block[n][AMR_NBR6]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), 0, (N2_GPU[n]) / (1 + REF_2), 0, D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send5_E, receive5_1E1, receive5_1E2, E,
					&(Bufferp[n]), &(Buffersend5E[block[block[n][AMR_NBR6]][AMR_CHILD2]]), &(Bufferrec5_1E1[n]), &(Bufferrec5_1E2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD2]][250]), calc_corr,
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][253], &Statbound[n][253]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_3E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive5_3E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive5_3E, receive5_3E1, receive5_3E2, E,
						&(Bufferp[n]), &(Bufferrec5_3E[n]), &(Bufferrec5_3E1[n]), &(Bufferrec5_3E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN11D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD4], block[block[n][AMR_NBR6]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send5_E, receive5_3E1, receive5_3E2, E,
						&(Bufferp[n]), &(Buffersend5E[block[block[n][AMR_NBR6]][AMR_CHILD4]]), &(Bufferrec5_3E1[n]), &(Bufferrec5_3E2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD4]][250]), calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN11D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][255], &Statbound[n][255]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_5E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive5_5E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive5_5E, receive5_5E1, receive5_5E2, E,
						&(Bufferp[n]), &(Bufferrec5_5E[n]), &(Bufferrec5_5E1[n]), &(Bufferrec5_5E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN5D_1] == -100)));
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD6], block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send5_E, receive5_5E1, receive5_5E2, E,
						&(Bufferp[n]), &(Buffersend5E[block[block[n][AMR_NBR6]][AMR_CHILD6]]), &(Bufferrec5_5E1[n]), &(Bufferrec5_5E2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD6]][250]), calc_corr,
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN5D_1] == -100)));
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][257], &Statbound[n][257]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_7E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive5_7E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive5_7E, receive5_7E1, receive5_7E2, E,
						&(Bufferp[n]), &(Bufferrec5_7E[n]), &(Bufferrec5_7E1[n]), &(Bufferrec5_7E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN11D_2] == -100)),
						block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN5D_2] == -100)));
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD8], block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send5_E, receive5_7E1, receive5_7E2, E,
						&(Bufferp[n]), &(Buffersend5E[block[block[n][AMR_NBR6]][AMR_CHILD8]]), &(Bufferrec5_7E1[n]), &(Bufferrec5_7E2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD8]][250]), calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN11D_2] == -100)),
						block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN5D_2] == -100)));
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][260], &Statbound[n][260]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N1_GPU[n] + N1G) *(N2_GPU[n] + N2G)*sizeof(double), receive6_E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_E(n, n, block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] + N1G, N2_GPU[n] + N2G, receive6_E, receive6_E1, NULL, E, &(Bufferp[n]), &(Bufferrec6E[n]), &(Bufferrec6E1[n]), NULL, NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR5] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) - 3 * (block[n][AMR_CORN9D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR5] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100),
					3 - (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR5] && block[n][AMR_CORN7D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) - 3 * (block[n][AMR_CORN7D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR5] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
			else{
				unpack_receive3_E(n, block[n][AMR_NBR5], block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] + N1G, N2_GPU[n] + N2G, send6_E, receive6_E1, NULL, E,
					&(Bufferp[n]), &(Buffersend6E[block[n][AMR_NBR5]]), &(Bufferrec6E1[n]), NULL, &(boundevent[block[n][AMR_NBR5]][260]), calc_corr,
					3 - (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR5] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) - 3 * (block[n][AMR_CORN9D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR5] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100),
					3 - (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR5] && block[n][AMR_CORN7D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) - 3 * (block[n][AMR_CORN7D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR5] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][262], &Statbound[n][262]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_2E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive6_2E[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive6_2E, receive6_2E1, receive6_2E2, E,
					&(Bufferp[n]), &(Bufferrec6_2E[n]), &(Bufferrec6_2E1[n]), &(Bufferrec6_2E2[n]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN9D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN7D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD1], block[block[n][AMR_NBR5]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send6_E, receive6_2E1, receive6_2E2, E,
					&(Bufferp[n]), &(Buffersend6E[block[block[n][AMR_NBR5]][AMR_CHILD1]]), &(Bufferrec6_2E1[n]), &(Bufferrec6_2E2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD1]][260]), calc_corr,
					1 - ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN9D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN7D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][264], &Statbound[n][264]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_4E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive6_4E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive6_4E, receive6_4E1, receive6_4E2, E,
						&(Bufferp[n]), &(Bufferrec6_4E[n]), &(Bufferrec6_4E1[n]), &(Bufferrec6_4E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN10D_1] == -100)),
						1 - ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN7D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD3], block[block[n][AMR_NBR5]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send6_E, receive6_4E1, receive6_4E2, E,
						&(Bufferp[n]), &(Buffersend6E[block[block[n][AMR_NBR5]][AMR_CHILD3]]), &(Bufferrec6_4E1[n]), &(Bufferrec6_4E2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD3]][260]), calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN10D_1] == -100)),
						1 - ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN7D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][266], &Statbound[n][266]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_6E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive6_6E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive6_6E, receive6_6E1, receive6_6E2, E,
						&(Bufferp[n]), &(Bufferrec6_6E[n]), &(Bufferrec6_6E1[n]), &(Bufferrec6_6E2[n]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN9D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN6D_1] == -100)));
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD5], block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send6_E, receive6_6E1, receive6_6E2, E,
						&(Bufferp[n]), &(Buffersend6E[block[block[n][AMR_NBR5]][AMR_CHILD5]]), &(Bufferrec6_6E1[n]), &(Bufferrec6_6E2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD5]][260]), calc_corr,
						1 - ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN9D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN6D_1] == -100)));
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][268], &Statbound[n][268]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_8E[n], CL_FALSE, (int)0 * sizeof(double), 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive6_8E[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), receive6_8E, receive6_8E1, receive6_8E2, E,
						&(Bufferp[n]), &(Bufferrec6_8E[n]), &(Bufferrec6_8E1[n]), &(Bufferrec6_8E2[n]), NULL, calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN10D_2] == -100)),
						block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN6D_2] == -100)));
				}
				else{
					unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD7], block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						(N1_GPU[n] + N1G) / (1 + REF_1), (N2_GPU[n] + N2G) / (1 + REF_2), send6_E, receive6_8E1, receive6_8E2, E,
						&(Bufferp[n]), &(Buffersend6E[block[block[n][AMR_NBR5]][AMR_CHILD7]]), &(Bufferrec6_8E1[n]), &(Bufferrec6_8E2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD7]][260]), calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN10D_2] == -100)),
						block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN6D_2] == -100)));
				}
			}
		}
	}
#endif
}


void E1_send_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
	if (block[n][AMR_CORN9] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == n || (block[n][AMR_CORN9D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN9]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN9], 0, N1_GPU[n] + D1, 0, N3_GPU[n], send_E1_corn9, E, &(Bufferp[n]), &(BuffersendE1corn9[n]),
				&(boundevent[n][459]));
			if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][459]);
					clReleaseEvent(boundevent[n][459]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn9[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + D1)*sizeof(double), send_E1_corn9[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E1_corn9[n][0], N1_GPU[n] + D1, MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (459 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == block[n][AMR_CORN9] || (block[n][AMR_CORN9D] == -100 && block[block[n][AMR_CORN9]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn11[n][0], (N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (461 * NB + block[n][AMR_CORN9]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][461]);
			}
		}
		if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_1] == block[block[n][AMR_CORN9]][AMR_CHILD3] || block[n][AMR_CORN9D_1] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn11_1[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE], (311 * NB + block[block[n][AMR_CORN9]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][311]);
		}
		if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_2] == block[block[n][AMR_CORN9]][AMR_CHILD7] || block[n][AMR_CORN9D_2] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn11_2[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE], (311 * NB + block[block[n][AMR_CORN9]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][361]);
		}
		if (block[block[n][AMR_CORN9]][AMR_PARENT] >= 0 && (block[n][AMR_CORN9D] == n || block[n][AMR_CORN9D] == -100)){
			if (block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN9] == block[block[n][AMR_CORN9]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN9]][AMR_PARENT], 0, N1_GPU[n] + N1G, 0, N3_GPU[n], send_E1_corn9, E,
					&(Bufferp[n]), &(BuffersendE1corn9[n]), &(boundevent[n][309]));
				if (block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][309]);
						clReleaseEvent(boundevent[n][309]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn9[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send_E1_corn9[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E1_corn9[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_NODE], (309 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[44]);
					MPI_Request_free(&req[44]);
				}
			}
		}
	}
	if (block[n][AMR_CORN10] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == n || (block[n][AMR_CORN10D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN10]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN10], 0, N1_GPU[n] + D1, N2_GPU[n], N3_GPU[n], send_E1_corn10, E, &(Bufferp[n]), &(BuffersendE1corn10[n]),
				&(boundevent[n][460]));
			if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][460]);
					clReleaseEvent(boundevent[n][460]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn10[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + D1)*sizeof(double), send_E1_corn10[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E1_corn10[n][0], N1_GPU[n] + D1, MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (460 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == block[n][AMR_CORN10] || (block[n][AMR_CORN10D] == -100 && block[block[n][AMR_CORN10]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn12[n][0], (N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (462 * NB + block[n][AMR_CORN10]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][462]);
			}
		}
		if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_1] == block[block[n][AMR_CORN10]][AMR_CHILD1] || block[n][AMR_CORN10D_1] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn12_1[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE], (312 * NB + block[block[n][AMR_CORN10]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][312]);
		}
		if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_2] == block[block[n][AMR_CORN10]][AMR_CHILD5] || block[n][AMR_CORN10D_2] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn12_2[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE], (312 * NB + block[block[n][AMR_CORN10]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][362]);
		}
		if (block[block[n][AMR_CORN10]][AMR_PARENT] >= 0 && (block[n][AMR_CORN10D] == n || block[n][AMR_CORN10D] == -100)){
			if (block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN10] == block[block[n][AMR_CORN10]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN10]][AMR_PARENT], 0, N1_GPU[n] + N1G, N2_GPU[n], N3_GPU[n], send_E1_corn10, E,
					&(Bufferp[n]), &(BuffersendE1corn10[n]), &(boundevent[n][310]));
				if (block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][310]);
						clReleaseEvent(boundevent[n][310]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn10[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send_E1_corn10[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E1_corn10[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_NODE], (310 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[45]);
					MPI_Request_free(&req[45]);
				}
			}
		}
	}
	if (block[n][AMR_CORN11] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == n || (block[n][AMR_CORN11D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN11]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN11], 0, N1_GPU[n] + D1, N2_GPU[n], 0, send_E1_corn11, E, &(Bufferp[n]), &(BuffersendE1corn11[n]),
				&(boundevent[n][461]));
			if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][461]);
					clReleaseEvent(boundevent[n][461]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn11[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + D1)*sizeof(double), send_E1_corn11[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E1_corn11[n][0], N1_GPU[n] + D1, MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (461 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == block[n][AMR_CORN11] || (block[n][AMR_CORN11D] == -100 && block[block[n][AMR_CORN11]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn9[n][0], (N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (459 * NB + block[n][AMR_CORN11]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][459]);
			}
		}
		if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_1] == block[block[n][AMR_CORN11]][AMR_CHILD2] || block[n][AMR_CORN11D_1] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn9_1[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE], (309 * NB + block[block[n][AMR_CORN11]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][309]);
		}
		if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_2] == block[block[n][AMR_CORN11]][AMR_CHILD6] || block[n][AMR_CORN11D_2] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn9_2[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE], (309 * NB + block[block[n][AMR_CORN11]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][359]);
		}
		if (block[block[n][AMR_CORN11]][AMR_PARENT] >= 0 && (block[n][AMR_CORN11D] == n || block[n][AMR_CORN11D] == -100)){
			if (block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN11] == block[block[n][AMR_CORN11]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN11]][AMR_PARENT], 0, N1_GPU[n] + N1G, N2_GPU[n], 0, send_E1_corn11, E,
					&(Bufferp[n]), &(BuffersendE1corn11[n]), &(boundevent[n][311]));
				if (block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][311]);
						clReleaseEvent(boundevent[n][311]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn11[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send_E1_corn11[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E1_corn11[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_NODE], (311 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[46]);
					MPI_Request_free(&req[46]);
				}
			}
		}
	}
	if (block[n][AMR_CORN12] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == n || (block[n][AMR_CORN12D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN12]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN12], 0, N1_GPU[n] + D1, 0, 0, send_E1_corn12, E, &(Bufferp[n]), &(BuffersendE1corn12[n]),
				&(boundevent[n][462]));
			if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][462]);
					clReleaseEvent(boundevent[n][462]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn12[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + D1)*sizeof(double), send_E1_corn12[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E1_corn12[n][0], N1_GPU[n] + D1, MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (462 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == block[n][AMR_CORN12] || (block[n][AMR_CORN12D] == -100 && block[block[n][AMR_CORN12]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn10[n][0], (N1_GPU[n] + N1G), MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (460 * NB + block[n][AMR_CORN12]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][460]);
			}
		}
		if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_1] == block[block[n][AMR_CORN12]][AMR_CHILD4] || block[n][AMR_CORN12D_1] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn10_1[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE], (310 * NB + block[block[n][AMR_CORN12]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][310]);
		}
		if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_2] == block[block[n][AMR_CORN12]][AMR_CHILD8] || block[n][AMR_CORN12D_2] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E1_corn10_2[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE], (310 * NB + block[block[n][AMR_CORN12]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][360]);
		}
		if (block[block[n][AMR_CORN12]][AMR_PARENT] >= 0 && (block[n][AMR_CORN12D] == n || block[n][AMR_CORN12D] == -100)){
			if (block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN12] == block[block[n][AMR_CORN12]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN12]][AMR_PARENT], 0, N1_GPU[n] + N1G, 0, 0, send_E1_corn12, E,
					&(Bufferp[n]), &(BuffersendE1corn12[n]), &(boundevent[n][312]));
				if (block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][312]);
						clReleaseEvent(boundevent[n][312]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE1corn12[n], CL_TRUE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), send_E1_corn12[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E1_corn12[n][0], (N1_GPU[n] + N1G) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_NODE], (312 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[47]);
					MPI_Request_free(&req[47]);
				}
			}
		}
	}
}

void E2_send_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
	if (block[n][AMR_CORN5] >= 0){
		if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == n || (block[n][AMR_CORN5D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN5]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN5], N1_GPU[n], 0, N2_GPU[n] + D2, 0, send_E2_corn5, E, &(Bufferp[n]), &(BuffersendE2corn5[n]),
				&(boundevent[n][455]));
			if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][455]);
					clReleaseEvent(boundevent[n][455]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn5[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + D2)*sizeof(double), send_E2_corn5[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E2_corn5[n][0], N2_GPU[n] + D2, MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (455 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == block[n][AMR_CORN5] || (block[n][AMR_CORN5D] == -100 && block[block[n][AMR_CORN5]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn7[n][0], (N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (457 * NB + block[n][AMR_CORN5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][457]);
			}
		}
		if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_1] == block[block[n][AMR_CORN5]][AMR_CHILD2] || block[n][AMR_CORN5D_1] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn7_1[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE], (307 * NB + block[block[n][AMR_CORN5]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][307]);
		}
		if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_2] == block[block[n][AMR_CORN5]][AMR_CHILD4] || block[n][AMR_CORN5D_2] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn7_2[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE], (307 * NB + block[block[n][AMR_CORN5]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][357]);
		}
		if (block[block[n][AMR_CORN5]][AMR_PARENT] >= 0 && (block[n][AMR_CORN5D] == n || block[n][AMR_CORN5D] == -100)){
			if (block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN5] == block[block[n][AMR_CORN5]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN5]][AMR_PARENT], N1_GPU[n], 0, N2_GPU[n] + N2G, 0, send_E2_corn5, E,
					&(Bufferp[n]), &(BuffersendE2corn5[n]), &(boundevent[n][305]));
				if (block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][305]);
						clReleaseEvent(boundevent[n][305]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn5[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send_E2_corn5[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E2_corn5[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_NODE], (305 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[48]);
					MPI_Request_free(&req[48]);
				}
			}
		}
	}
	if (block[n][AMR_CORN6] >= 0){
		if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == n || (block[n][AMR_CORN6D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN6]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN6], N1_GPU[n], 0, N2_GPU[n] + D2, N3_GPU[n], send_E2_corn6, E, &(Bufferp[n]), &(BuffersendE2corn6[n]),
				&(boundevent[n][456]));
			if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][456]);
					clReleaseEvent(boundevent[n][456]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn6[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + D2)*sizeof(double), send_E2_corn6[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E2_corn6[n][0], N2_GPU[n] + D2, MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (456 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == block[n][AMR_CORN6] || (block[n][AMR_CORN6D] == -100 && block[block[n][AMR_CORN6]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn8[n][0], (N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (458 * NB + block[n][AMR_CORN6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][458]);
			}
		}
		if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_1] == block[block[n][AMR_CORN6]][AMR_CHILD1] || block[n][AMR_CORN6D_1] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn8_1[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE], (308 * NB + block[block[n][AMR_CORN6]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][308]);
		}
		if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_2] == block[block[n][AMR_CORN6]][AMR_CHILD3] || block[n][AMR_CORN6D_2] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn8_2[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE], (308 * NB + block[block[n][AMR_CORN6]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][358]);
		}
		if (block[block[n][AMR_CORN6]][AMR_PARENT] >= 0 && (block[n][AMR_CORN6D] == n || block[n][AMR_CORN6D] == -100)){
			if (block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN6] == block[block[n][AMR_CORN6]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN6]][AMR_PARENT], N1_GPU[n], 0, N2_GPU[n] + N2G, N3_GPU[n], send_E2_corn6, E,
					&(Bufferp[n]), &(BuffersendE2corn6[n]), &(boundevent[n][306]));
				if (block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][306]);
						clReleaseEvent(boundevent[n][306]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn6[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send_E2_corn6[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E2_corn6[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_NODE], (306 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[49]);
					MPI_Request_free(&req[49]);
				}
			}
		}
	}
	if (block[n][AMR_CORN7] >= 0){
		if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == n || (block[n][AMR_CORN7D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN7]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN7], 0, 0, N2_GPU[n] + D2, N3_GPU[n], send_E2_corn7, E, &(Bufferp[n]), &(BuffersendE2corn7[n]),
				&(boundevent[n][457]));
			if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][457]);
					clReleaseEvent(boundevent[n][457]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn7[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + D2)*sizeof(double), send_E2_corn7[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E2_corn7[n][0], N2_GPU[n] + D2, MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (457 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == block[n][AMR_CORN7] || (block[n][AMR_CORN7D] == -100 && block[block[n][AMR_CORN7]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn5[n][0], (N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (455 * NB + block[n][AMR_CORN7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][455]);
			}
		}
		if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_1] == block[block[n][AMR_CORN7]][AMR_CHILD5] || block[n][AMR_CORN7D_1] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn5_1[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE], (305 * NB + block[block[n][AMR_CORN7]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][305]);
		}
		if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_2] == block[block[n][AMR_CORN7]][AMR_CHILD7] || block[n][AMR_CORN7D_2] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn5_2[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE], (305 * NB + block[block[n][AMR_CORN7]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][355]);
		}

		if (block[block[n][AMR_CORN7]][AMR_PARENT] >= 0 && (block[n][AMR_CORN7D] == n || block[n][AMR_CORN7D] == -100)){
			if (block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN7] == block[block[n][AMR_CORN7]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN7]][AMR_PARENT], 0, 0, N2_GPU[n] + N2G, N3_GPU[n], send_E2_corn7, E,
					&(Bufferp[n]), &(BuffersendE2corn7[n]), &(boundevent[n][307]));
				if (block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][307]);
						clReleaseEvent(boundevent[n][307]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn7[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send_E2_corn7[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E2_corn7[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_NODE], (307 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[50]);
					MPI_Request_free(&req[50]);
				}
			}
		}
	}
	if (block[n][AMR_CORN8] >= 0){
		if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == n || (block[n][AMR_CORN8D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN8]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN8], 0, 0, N2_GPU[n] + D2, 0, send_E2_corn8, E, &(Bufferp[n]), &(BuffersendE2corn8[n]),
				&(boundevent[n][458]));
			if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][458]);
					clReleaseEvent(boundevent[n][458]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn8[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + D2)*sizeof(double), send_E2_corn8[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E2_corn8[n][0], N2_GPU[n] + D2, MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (458 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == block[n][AMR_CORN8] || (block[n][AMR_CORN8D] == -100 && block[block[n][AMR_CORN8]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn6[n][0], (N2_GPU[n] + N2G), MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (456 * NB + block[n][AMR_CORN8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][456]);
			}
		}
		if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_1] == block[block[n][AMR_CORN8]][AMR_CHILD6] || block[n][AMR_CORN8D_1] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn6_1[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE], (306 * NB + block[block[n][AMR_CORN8]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][306]);
		}
		if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_2] == block[block[n][AMR_CORN8]][AMR_CHILD8] || block[n][AMR_CORN8D_2] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E2_corn6_2[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE], (306 * NB + block[block[n][AMR_CORN8]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][356]);
		}
		if (block[block[n][AMR_CORN8]][AMR_PARENT] >= 0 && (block[n][AMR_CORN8D] == n || block[n][AMR_CORN8D] == -100)){
			if (block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN8] == block[block[n][AMR_CORN8]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN8]][AMR_PARENT], 0, 0, N2_GPU[n] + N2G, 0, send_E2_corn8, E,
					&(Bufferp[n]), &(BuffersendE2corn8[n]), &(boundevent[n][308]));
				if (block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][308]);
						clReleaseEvent(boundevent[n][308]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE2corn8[n], CL_TRUE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), send_E2_corn8[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E2_corn8[n][0], (N2_GPU[n] + N2G) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_NODE], (308 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[51]);
					MPI_Request_free(&req[51]);
				}
			}
		}
	}
}

void E3_send_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n){
	if (block[n][AMR_CORN1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == n || (block[n][AMR_CORN1D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN1]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN1], N1_GPU[n], 0, 0, N3_GPU[n] + D3, send_E3_corn1, E, &(Bufferp[n]), &(BuffersendE3corn1[n]),
				&(boundevent[n][451]));
			if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][451]);
					clReleaseEvent(boundevent[n][451]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn1[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + D3)*sizeof(double), send_E3_corn1[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E3_corn1[n][0], N3_GPU[n] + D3, MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (451 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == block[n][AMR_CORN1] || (block[n][AMR_CORN1D] == -100 && block[block[n][AMR_CORN1]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn3[n][0], (N3_GPU[n] + N3G), MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (453 * NB + block[n][AMR_CORN1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][453]);
			}
		}
		if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_1] == block[block[n][AMR_CORN1]][AMR_CHILD3] || block[n][AMR_CORN1D_1] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn3_1[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE], (303 * NB + block[block[n][AMR_CORN1]][AMR_CHILD3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][303]);
		}
		if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_2] == block[block[n][AMR_CORN1]][AMR_CHILD4] || block[n][AMR_CORN1D_2] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn3_2[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE], (303 * NB + block[block[n][AMR_CORN1]][AMR_CHILD4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][353]);
		}
		if (block[block[n][AMR_CORN1]][AMR_PARENT] >= 0 && (block[n][AMR_CORN1D] == n || block[n][AMR_CORN1D] == -100)){
			if (block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN1] == block[block[n][AMR_CORN1]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN1]][AMR_PARENT], N1_GPU[n], 0, 0, N3_GPU[n] + N3G, send_E3_corn1, E,
					&(Bufferp[n]), &(BuffersendE3corn1[n]), &(boundevent[n][301]));
				if (block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][301]);
						clReleaseEvent(boundevent[n][301]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn1[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), send_E3_corn1[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E3_corn1[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_NODE], (301 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[52]);
					MPI_Request_free(&req[52]);
				}
			}
		}
	}
	if (block[n][AMR_CORN2] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == n || (block[n][AMR_CORN2D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN2]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN2], N1_GPU[n], N2_GPU[n], 0, N3_GPU[n] + D3, send_E3_corn2, E, &(Bufferp[n]), &(BuffersendE3corn2[n]),
				&(boundevent[n][452]));
			if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][452]);
					clReleaseEvent(boundevent[n][452]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn2[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + D3)*sizeof(double), send_E3_corn2[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E3_corn2[n][0], N3_GPU[n] + D3, MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (452 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == block[n][AMR_CORN2] || (block[n][AMR_CORN2D] == -100 && block[block[n][AMR_CORN2]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn4[n][0], (N3_GPU[n] + N3G), MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (454 * NB + block[n][AMR_CORN2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][454]);
			}
		}
		if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_1] == block[block[n][AMR_CORN2]][AMR_CHILD1] || block[n][AMR_CORN2D_1] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn4_1[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE], (304 * NB + block[block[n][AMR_CORN2]][AMR_CHILD1]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][304]);
		}
		if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_2] == block[block[n][AMR_CORN2]][AMR_CHILD2] || block[n][AMR_CORN2D_2] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn4_2[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE], (304 * NB + block[block[n][AMR_CORN2]][AMR_CHILD2]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][354]);
		}
		if (block[block[n][AMR_CORN2]][AMR_PARENT] >= 0 && (block[n][AMR_CORN2D] == n || block[n][AMR_CORN2D] == -100)){
			if (block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN2] == block[block[n][AMR_CORN2]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN2]][AMR_PARENT], N1_GPU[n], N2_GPU[n], 0, N3_GPU[n] + N3G, send_E3_corn2, E,
					&(Bufferp[n]), &(BuffersendE3corn2[n]), &(boundevent[n][302]));
				if (block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][302]);
						clReleaseEvent(boundevent[n][302]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn2[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), send_E3_corn2[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E3_corn2[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_NODE], (302 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[53]);
					MPI_Request_free(&req[53]);
				}
			}
		}
	}
	if (block[n][AMR_CORN3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == n || (block[n][AMR_CORN3D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN3]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN3], 0, N2_GPU[n], 0, N3_GPU[n] + D3, send_E3_corn3, E, &(Bufferp[n]), &(BuffersendE3corn3[n]),
				&(boundevent[n][453]));
			if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][453]);
					clReleaseEvent(boundevent[n][453]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn3[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + D3)*sizeof(double), send_E3_corn3[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E3_corn3[n][0], N3_GPU[n] + D3, MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (453 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == block[n][AMR_CORN3] || (block[n][AMR_CORN3D] == -100 && block[block[n][AMR_CORN3]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn1[n][0], (N3_GPU[n] + N3G), MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (451 * NB + block[n][AMR_CORN3]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][451]);
			}
		}
		if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_1] == block[block[n][AMR_CORN3]][AMR_CHILD5] || block[n][AMR_CORN3D_1] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn1_1[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE], (301 * NB + block[block[n][AMR_CORN3]][AMR_CHILD5]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][301]);
		}
		if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_2] == block[block[n][AMR_CORN3]][AMR_CHILD6] || block[n][AMR_CORN3D_2] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn1_2[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE], (301 * NB + block[block[n][AMR_CORN3]][AMR_CHILD6]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][351]);
		}
		if (block[block[n][AMR_CORN3]][AMR_PARENT] >= 0 && (block[n][AMR_CORN3D] == n || block[n][AMR_CORN3D] == -100)){
			if (block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN3] == block[block[n][AMR_CORN3]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN3]][AMR_PARENT], 0, N2_GPU[n], 0, N3_GPU[n] + N3G, send_E3_corn3, E,
					&(Bufferp[n]), &(BuffersendE3corn3[n]), &(boundevent[n][303]));
				if (block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][303]);
						clReleaseEvent(boundevent[n][303]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn3[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), send_E3_corn3[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E3_corn3[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_NODE], (303 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[54]);
					MPI_Request_free(&req[54]);
				}
			}
		}
	}
	if (block[n][AMR_CORN4] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == n || (block[n][AMR_CORN4D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN4]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN4], 0, 0, 0, N3_GPU[n] + D3, send_E3_corn4, E, &(Bufferp[n]), &(BuffersendE3corn4[n]),
				&(boundevent[n][454]));
			if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][454]);
					clReleaseEvent(boundevent[n][454]);
					clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn4[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + D3)*sizeof(double), send_E3_corn4[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send_E3_corn4[n][0], N3_GPU[n] + D3, MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (454 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == block[n][AMR_CORN4] || (block[n][AMR_CORN4D] == -100 && block[block[n][AMR_CORN4]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn2[n][0], (N3_GPU[n] + N3G), MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (452 * NB + block[n][AMR_CORN4]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][452]);
			}
		}
		if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_1] == block[block[n][AMR_CORN4]][AMR_CHILD7] || block[n][AMR_CORN4D_1] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn2_1[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE], (302 * NB + block[block[n][AMR_CORN4]][AMR_CHILD7]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][302]);
		}
		if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_2] == block[block[n][AMR_CORN4]][AMR_CHILD8] || block[n][AMR_CORN4D_2] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive_E3_corn2_2[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE], (302 * NB + block[block[n][AMR_CORN4]][AMR_CHILD8]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n][352]);
		}
		if (block[block[n][AMR_CORN4]][AMR_PARENT] >= 0 && (block[n][AMR_CORN4D] == n || block[n][AMR_CORN4D] == -100)){
			if (block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN4] == block[block[n][AMR_CORN4]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN4]][AMR_PARENT], 0, 0, 0, N3_GPU[n] + N3G, send_E3_corn4, E,
					&(Bufferp[n]), &(BuffersendE3corn4[n]), &(boundevent[n][304]));
				if (block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][304]);
						clReleaseEvent(boundevent[n][304]);
						clEnqueueReadBuffer(commandQueueGPU[n], BuffersendE3corn4[n], CL_TRUE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), send_E3_corn4[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send_E3_corn4[n][0], (N3_GPU[n] + N3G) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_NODE], (304 * NB + n) % MPI_TAG_MAX, mpi_cartcomm, &req[55]);
					MPI_Request_free(&req[55]);
				}
			}
		}
	}
}

/*Receive boundaries for compute nodes through MPI*/
void E1_receive_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_CORN9] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == block[n][AMR_CORN9] || (block[n][AMR_CORN9D] == -100 && block[block[n][AMR_CORN9]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][461], &Statbound[n][461]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn11[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + D1) *sizeof(double), receive_E1_corn11[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN9], 0, N1_GPU[n], 0, N3_GPU[n], receive_E1_corn11, tempreceive_E1_corn11, NULL, E,
					&(Bufferp[n]), &(BufferrecE1corn11[n]), &(tempBufferrecE1corn11[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN9], block[n][AMR_CORN9], 0, N1_GPU[n], 0, N3_GPU[n], send_E1_corn11, receive_E1_corn11, NULL, E,
					&(Bufferp[n]), &(BuffersendE1corn11[block[n][AMR_CORN9]]), &(BufferrecE1corn11[n]), NULL, &(boundevent[block[n][AMR_CORN9]][461]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN9D_1] == block[block[n][AMR_CORN9]][AMR_CHILD3] || block[n][AMR_CORN9D_1] == -100){
				if (block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][311], &Statbound[n][311]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn11_2[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn11_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN9]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), 0, N3_GPU[n], receive_E1_corn11_1, tempreceive_E1_corn11_1, receive_E1_corn11_12, E,
						&(Bufferp[n]), &(BufferrecE1corn11_2[n]), &(tempBufferrecE1corn11_2[n]), &(BufferrecE1corn11_22[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN9]][AMR_CHILD3], block[block[n][AMR_CORN9]][AMR_CHILD3], 0, (N1_GPU[n]) / (1 + REF_1), 0, N3_GPU[n], send_E1_corn11, receive_E1_corn11_1, receive_E1_corn11_12, E,
						&(Bufferp[n]), &(BuffersendE1corn11[block[block[n][AMR_CORN9]][AMR_CHILD3]]), &(BufferrecE1corn11_2[n]), &(BufferrecE1corn11_22[n]), &(boundevent[block[block[n][AMR_CORN9]][AMR_CHILD3]][311]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN9D_2] == block[block[n][AMR_CORN9]][AMR_CHILD7] || block[n][AMR_CORN9D_2] == -100)){
				if (block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][361], &Statbound[n][361]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn11_6[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn11_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN9]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N3_GPU[n], receive_E1_corn11_2, tempreceive_E1_corn11_2, receive_E1_corn11_22, E,
						&(Bufferp[n]), &(BufferrecE1corn11_6[n]), &(tempBufferrecE1corn11_6[n]), &(BufferrecE1corn11_62[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN9]][AMR_CHILD7], block[block[n][AMR_CORN9]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N3_GPU[n], send_E1_corn11, receive_E1_corn11_2, receive_E1_corn11_22, E,
						&(Bufferp[n]), &(BuffersendE1corn11[block[block[n][AMR_CORN9]][AMR_CHILD7]]), &(BufferrecE1corn11_6[n]), &(BufferrecE1corn11_62[n]), &(boundevent[block[block[n][AMR_CORN9]][AMR_CHILD7]][311]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN10] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == block[n][AMR_CORN10] || (block[n][AMR_CORN10D] == -100 && block[block[n][AMR_CORN10]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][462], &Statbound[n][462]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn12[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + D1) *sizeof(double), receive_E1_corn12[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN10], 0, N1_GPU[n], N2_GPU[n], N3_GPU[n], receive_E1_corn12, tempreceive_E1_corn12, NULL, E,
					&(Bufferp[n]), &(BufferrecE1corn12[n]), &(tempBufferrecE1corn12[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN10], block[n][AMR_CORN10], 0, N1_GPU[n], N2_GPU[n], N3_GPU[n], send_E1_corn12, receive_E1_corn12, NULL, E,
					&(Bufferp[n]), &(BuffersendE1corn12[block[n][AMR_CORN10]]), &(BufferrecE1corn12[n]), NULL, &(boundevent[block[n][AMR_CORN10]][462]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN10D_1] == block[block[n][AMR_CORN10]][AMR_CHILD1] || block[n][AMR_CORN10D_1] == -100){
				if (block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][312], &Statbound[n][312]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn12_4[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn12_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN10]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N3_GPU[n], receive_E1_corn12_1, tempreceive_E1_corn12_1, receive_E1_corn12_12, E,
						&(Bufferp[n]), &(BufferrecE1corn12_4[n]), &(tempBufferrecE1corn12_4[n]), &(BufferrecE1corn12_42[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN10]][AMR_CHILD1], block[block[n][AMR_CORN10]][AMR_CHILD1], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], N3_GPU[n], send_E1_corn12, receive_E1_corn12_1, receive_E1_corn12_12, E,
						&(Bufferp[n]), &(BuffersendE1corn12[block[block[n][AMR_CORN10]][AMR_CHILD1]]), &(BufferrecE1corn12_4[n]), &(BufferrecE1corn12_42[n]), &(boundevent[block[block[n][AMR_CORN10]][AMR_CHILD1]][312]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN10D_2] == block[block[n][AMR_CORN10]][AMR_CHILD5] || block[n][AMR_CORN10D_2] == -100)){
				if (block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][362], &Statbound[n][362]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn12_8[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn12_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN10]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N3_GPU[n], receive_E1_corn12_2, tempreceive_E1_corn12_2, receive_E1_corn12_22, E,
						&(Bufferp[n]), &(BufferrecE1corn12_8[n]), &(tempBufferrecE1corn12_8[n]), &(BufferrecE1corn12_82[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN10]][AMR_CHILD5], block[block[n][AMR_CORN10]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N3_GPU[n], send_E1_corn12, receive_E1_corn12_2, receive_E1_corn12_22, E,
						&(Bufferp[n]), &(BuffersendE1corn12[block[block[n][AMR_CORN10]][AMR_CHILD5]]), &(BufferrecE1corn12_8[n]), &(BufferrecE1corn12_82[n]), &(boundevent[block[block[n][AMR_CORN10]][AMR_CHILD5]][312]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN11] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == block[n][AMR_CORN11] || (block[n][AMR_CORN11D] == -100 && block[block[n][AMR_CORN11]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][459], &Statbound[n][459]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn9[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + D1) *sizeof(double), receive_E1_corn9[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN11], 0, N1_GPU[n], N2_GPU[n], 0, receive_E1_corn9, tempreceive_E1_corn9, NULL, E,
					&(Bufferp[n]), &(BufferrecE1corn9[n]), &(tempBufferrecE1corn9[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN11], block[n][AMR_CORN11], 0, N1_GPU[n], N2_GPU[n], 0, send_E1_corn9, receive_E1_corn9, NULL, E,
					&(Bufferp[n]), &(BuffersendE1corn9[block[n][AMR_CORN11]]), &(BufferrecE1corn9[n]), NULL, &(boundevent[block[n][AMR_CORN11]][459]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN11D_1] == block[block[n][AMR_CORN11]][AMR_CHILD2] || block[n][AMR_CORN11D_1] == -100){
				if (block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][309], &Statbound[n][309]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn9_3[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn9_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN11]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], 0, receive_E1_corn9_1, tempreceive_E1_corn9_1, receive_E1_corn9_12, E,
						&(Bufferp[n]), &(BufferrecE1corn9_3[n]), &(tempBufferrecE1corn9_3[n]), &(BufferrecE1corn9_32[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN11]][AMR_CHILD2], block[block[n][AMR_CORN11]][AMR_CHILD2], 0, (N1_GPU[n]) / (1 + REF_1), N2_GPU[n], 0, send_E1_corn9, receive_E1_corn9_1, receive_E1_corn9_12, E,
						&(Bufferp[n]), &(BuffersendE1corn9[block[block[n][AMR_CORN11]][AMR_CHILD2]]), &(BufferrecE1corn9_3[n]), &(BufferrecE1corn9_32[n]), &(boundevent[block[block[n][AMR_CORN11]][AMR_CHILD2]][309]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN11D_2] == block[block[n][AMR_CORN11]][AMR_CHILD6] || block[n][AMR_CORN11D_2] == -100)){
				if (block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][359], &Statbound[n][359]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn9_7[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn9_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN11]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], 0, receive_E1_corn9_2, tempreceive_E1_corn9_2, receive_E1_corn9_22, E,
						&(Bufferp[n]), &(BufferrecE1corn9_7[n]), &(tempBufferrecE1corn9_7[n]), &(BufferrecE1corn9_72[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN11]][AMR_CHILD6], block[block[n][AMR_CORN11]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], 0, send_E1_corn9, receive_E1_corn9_2, receive_E1_corn9_22, E,
						&(Bufferp[n]), &(BuffersendE1corn9[block[block[n][AMR_CORN11]][AMR_CHILD6]]), &(BufferrecE1corn9_7[n]), &(BufferrecE1corn9_72[n]), &(boundevent[block[block[n][AMR_CORN11]][AMR_CHILD6]][309]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN12] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == block[n][AMR_CORN12] || (block[n][AMR_CORN12D] == -100 && block[block[n][AMR_CORN12]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][460], &Statbound[n][460]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn10[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + D1) *sizeof(double), receive_E1_corn10[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN12], 0, N1_GPU[n], 0, 0, receive_E1_corn10, tempreceive_E1_corn10, NULL, E,
					&(Bufferp[n]), &(BufferrecE1corn10[n]), &(tempBufferrecE1corn10[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN12], block[n][AMR_CORN12], 0, N1_GPU[n], 0, 0, send_E1_corn10, receive_E1_corn10, NULL, E,
					&(Bufferp[n]), &(BuffersendE1corn10[block[n][AMR_CORN12]]), &(BufferrecE1corn10[n]), NULL, &(boundevent[block[n][AMR_CORN12]][460]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
			//receive from finer grid
			if (block[n][AMR_CORN12D_1] == block[block[n][AMR_CORN12]][AMR_CHILD4] || block[n][AMR_CORN12D_1] == -100){
				if (block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][310], &Statbound[n][310]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn10_1[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn10_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN12]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), 0, 0, receive_E1_corn10_1, tempreceive_E1_corn10_1, receive_E1_corn10_12, E,
						&(Bufferp[n]), &(BufferrecE1corn10_1[n]), &(tempBufferrecE1corn10_1[n]), &(BufferrecE1corn10_12[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN12]][AMR_CHILD4], block[block[n][AMR_CORN12]][AMR_CHILD4], 0, (N1_GPU[n]) / (1 + REF_1), 0, 0, send_E1_corn10, receive_E1_corn10_1, receive_E1_corn10_12, E,
						&(Bufferp[n]), &(BuffersendE1corn10[block[block[n][AMR_CORN12]][AMR_CHILD4]]), &(BufferrecE1corn10_1[n]), &(BufferrecE1corn10_12[n]), &(boundevent[block[block[n][AMR_CORN12]][AMR_CHILD4]][310]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN12D_2] == block[block[n][AMR_CORN12]][AMR_CHILD8] || block[n][AMR_CORN12D_2] == -100)){
				if (block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][360], &Statbound[n][360]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE1corn10_5[n], CL_FALSE, (int)0 * sizeof(double), (N1_GPU[n] + N1G) / (1 + REF_1)*sizeof(double), receive_E1_corn10_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN12]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 0, receive_E1_corn10_2, tempreceive_E1_corn10_2, receive_E1_corn10_22, E,
						&(Bufferp[n]), &(BufferrecE1corn10_5[n]), &(tempBufferrecE1corn10_5[n]), &(BufferrecE1corn10_52[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN12]][AMR_CHILD8], block[block[n][AMR_CORN12]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 0, send_E1_corn10, receive_E1_corn10_2, receive_E1_corn10_22, E,
						&(Bufferp[n]), &(BuffersendE1corn10[block[block[n][AMR_CORN12]][AMR_CHILD8]]), &(BufferrecE1corn10_5[n]), &(BufferrecE1corn10_52[n]), &(boundevent[block[block[n][AMR_CORN12]][AMR_CHILD8]][310]), calc_corr);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E2_receive_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_CORN5] >= 0){
		if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == block[n][AMR_CORN5] || (block[n][AMR_CORN5D] == -100 && block[block[n][AMR_CORN5]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][457], &Statbound[n][457]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn7[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + D2) *sizeof(double), receive_E2_corn7[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN5], N1_GPU[n], 0, N2_GPU[n], 0, receive_E2_corn7, tempreceive_E2_corn7, NULL, E,
					&(Bufferp[n]), &(BufferrecE2corn7[n]), &(tempBufferrecE2corn7[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN5], block[n][AMR_CORN5], N1_GPU[n], 0, N2_GPU[n], 0, send_E2_corn7, receive_E2_corn7, NULL, E,
					&(Bufferp[n]), &(BuffersendE2corn7[block[n][AMR_CORN5]]), &(BufferrecE2corn7[n]), NULL, &(boundevent[block[n][AMR_CORN5]][457]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN5D_1] == block[block[n][AMR_CORN5]][AMR_CHILD2] || block[n][AMR_CORN5D_1] == -100){
				if (block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][307], &Statbound[n][307]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn7_5[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn7_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN5]][AMR_CHILD2], N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), 0, receive_E2_corn7_1, tempreceive_E2_corn7_1, receive_E2_corn7_12, E,
						&(Bufferp[n]), &(BufferrecE2corn7_5[n]), &(tempBufferrecE2corn7_5[n]), &(BufferrecE2corn7_52[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN5]][AMR_CHILD2], block[block[n][AMR_CORN5]][AMR_CHILD2], N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), 0, send_E2_corn7, receive_E2_corn7_1, receive_E2_corn7_12, E,
						&(Bufferp[n]), &(BuffersendE2corn7[block[block[n][AMR_CORN5]][AMR_CHILD2]]), &(BufferrecE2corn7_5[n]), &(BufferrecE2corn7_52[n]), &(boundevent[block[block[n][AMR_CORN5]][AMR_CHILD2]][307]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN5D_2] == block[block[n][AMR_CORN5]][AMR_CHILD4] || block[n][AMR_CORN5D_2] == -100)){
				if (block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][357], &Statbound[n][357]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn7_7[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn7_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN5]][AMR_CHILD4], N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, receive_E2_corn7_2, tempreceive_E2_corn7_2, receive_E2_corn7_22, E,
						&(Bufferp[n]), &(BufferrecE2corn7_7[n]), &(tempBufferrecE2corn7_7[n]), &(BufferrecE2corn7_72[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN5]][AMR_CHILD4], block[block[n][AMR_CORN5]][AMR_CHILD4], N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, send_E2_corn7, receive_E2_corn7_2, receive_E2_corn7_22, E,
						&(Bufferp[n]), &(BuffersendE2corn7[block[block[n][AMR_CORN5]][AMR_CHILD4]]), &(BufferrecE2corn7_7[n]), &(BufferrecE2corn7_72[n]), &(boundevent[block[block[n][AMR_CORN5]][AMR_CHILD4]][307]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN6] >= 0){
		if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == block[n][AMR_CORN6] || (block[n][AMR_CORN6D] == -100 && block[block[n][AMR_CORN6]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][458], &Statbound[n][458]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn8[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + D2) *sizeof(double), receive_E2_corn8[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN6], N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], receive_E2_corn8, tempreceive_E2_corn8, NULL, E,
					&(Bufferp[n]), &(BufferrecE2corn8[n]), &(tempBufferrecE2corn8[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN6], block[n][AMR_CORN6], N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], send_E2_corn8, receive_E2_corn8, NULL, E,
					&(Bufferp[n]), &(BuffersendE2corn8[block[n][AMR_CORN6]]), &(BufferrecE2corn8[n]), NULL, &(boundevent[block[n][AMR_CORN6]][458]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN6D_1] == block[block[n][AMR_CORN6]][AMR_CHILD1] || block[n][AMR_CORN6D_1] == -100){
				if (block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][308], &Statbound[n][308]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn8_6[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn8_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN6]][AMR_CHILD1], N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], receive_E2_corn8_1, tempreceive_E2_corn8_1, receive_E2_corn8_12, E,
						&(Bufferp[n]), &(BufferrecE2corn8_6[n]), &(tempBufferrecE2corn8_6[n]), &(BufferrecE2corn8_62[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN6]][AMR_CHILD1], block[block[n][AMR_CORN6]][AMR_CHILD1], N1_GPU[n], 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], send_E2_corn8, receive_E2_corn8_1, receive_E2_corn8_12, E,
						&(Bufferp[n]), &(BuffersendE2corn8[block[block[n][AMR_CORN6]][AMR_CHILD1]]), &(BufferrecE2corn8_6[n]), &(BufferrecE2corn8_62[n]), &(boundevent[block[block[n][AMR_CORN6]][AMR_CHILD1]][308]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN6D_2] == block[block[n][AMR_CORN6]][AMR_CHILD3] || block[n][AMR_CORN6D_2] == -100)){
				if (block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][358], &Statbound[n][358]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn8_8[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn8_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN6]][AMR_CHILD3], N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], receive_E2_corn8_2, tempreceive_E2_corn8_2, receive_E2_corn8_22, E,
						&(Bufferp[n]), &(BufferrecE2corn8_8[n]), &(tempBufferrecE2corn8_8[n]), &(BufferrecE2corn8_82[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN6]][AMR_CHILD3], block[block[n][AMR_CORN6]][AMR_CHILD3], N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], send_E2_corn8, receive_E2_corn8_2, receive_E2_corn8_22, E,
						&(Bufferp[n]), &(BuffersendE2corn8[block[block[n][AMR_CORN6]][AMR_CHILD3]]), &(BufferrecE2corn8_8[n]), &(BufferrecE2corn8_82[n]), &(boundevent[block[block[n][AMR_CORN6]][AMR_CHILD3]][308]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN7] >= 0){
		if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == block[n][AMR_CORN7] || (block[n][AMR_CORN7D] == -100 && block[block[n][AMR_CORN7]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][455], &Statbound[n][455]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn5[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + D2) *sizeof(double), receive_E2_corn5[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN7], 0, 0, N2_GPU[n], N3_GPU[n], receive_E2_corn5, tempreceive_E2_corn5, NULL, E,
					&(Bufferp[n]), &(BufferrecE2corn5[n]), &(tempBufferrecE2corn5[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN7], block[n][AMR_CORN7], 0, 0, N2_GPU[n], N3_GPU[n], send_E2_corn5, receive_E2_corn5, NULL, E,
					&(Bufferp[n]), &(BuffersendE2corn5[block[n][AMR_CORN7]]), &(BufferrecE2corn5[n]), NULL, &(boundevent[block[n][AMR_CORN7]][455]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN7D_1] == block[block[n][AMR_CORN7]][AMR_CHILD5] || block[n][AMR_CORN7D_1] == -100){
				if (block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][305], &Statbound[n][305]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn5_2[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn5_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN7]][AMR_CHILD5], 0, 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], receive_E2_corn5_1, tempreceive_E2_corn5_1, receive_E2_corn5_12, E,
						&(Bufferp[n]), &(BufferrecE2corn5_2[n]), &(tempBufferrecE2corn5_2[n]), &(BufferrecE2corn5_22[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN7]][AMR_CHILD5], block[block[n][AMR_CORN7]][AMR_CHILD5], 0, 0, (N2_GPU[n]) / (1 + REF_2), N3_GPU[n], send_E2_corn5, receive_E2_corn5_1, receive_E2_corn5_12, E,
						&(Bufferp[n]), &(BuffersendE2corn5[block[block[n][AMR_CORN7]][AMR_CHILD5]]), &(BufferrecE2corn5_2[n]), &(BufferrecE2corn5_22[n]), &(boundevent[block[block[n][AMR_CORN7]][AMR_CHILD5]][305]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN7D_2] == block[block[n][AMR_CORN7]][AMR_CHILD7] || block[n][AMR_CORN7D_2] == -100)){
				if (block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][355], &Statbound[n][355]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn5_4[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn5_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN7]][AMR_CHILD7], 0, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], receive_E2_corn5_2, tempreceive_E2_corn5_2, receive_E2_corn5_22, E,
						&(Bufferp[n]), &(BufferrecE2corn5_4[n]), &(tempBufferrecE2corn5_4[n]), &(BufferrecE2corn5_42[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN7]][AMR_CHILD7], block[block[n][AMR_CORN7]][AMR_CHILD7], 0, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], send_E2_corn5, receive_E2_corn5_2, receive_E2_corn5_22, E,
						&(Bufferp[n]), &(BuffersendE2corn5[block[block[n][AMR_CORN7]][AMR_CHILD7]]), &(BufferrecE2corn5_4[n]), &(BufferrecE2corn5_42[n]), &(boundevent[block[block[n][AMR_CORN7]][AMR_CHILD7]][305]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN8] >= 0){
		if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == block[n][AMR_CORN8] || (block[n][AMR_CORN8D] == -100 && block[block[n][AMR_CORN8]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][456], &Statbound[n][456]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn6[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + D2) *sizeof(double), receive_E2_corn6[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN8], 0, 0, N2_GPU[n], 0, receive_E2_corn6, tempreceive_E2_corn6, NULL, E,
					&(Bufferp[n]), &(BufferrecE2corn6[n]), &(tempBufferrecE2corn6[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN8], block[n][AMR_CORN8], 0, 0, N2_GPU[n], 0, send_E2_corn6, receive_E2_corn6, NULL, E,
					&(Bufferp[n]), &(BuffersendE2corn6[block[n][AMR_CORN8]]), &(BufferrecE2corn6[n]), NULL, &(boundevent[block[n][AMR_CORN8]][456]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN8D_1] == block[block[n][AMR_CORN8]][AMR_CHILD6] || block[n][AMR_CORN8D_1] == -100){
				if (block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][306], &Statbound[n][306]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn6_1[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn6_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN8]][AMR_CHILD6], 0, 0, (N2_GPU[n]) / (1 + REF_2), 0, receive_E2_corn6_1, tempreceive_E2_corn6_1, receive_E2_corn6_12, E,
						&(Bufferp[n]), &(BufferrecE2corn6_1[n]), &(tempBufferrecE2corn6_1[n]), &(BufferrecE2corn6_12[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN8]][AMR_CHILD6], block[block[n][AMR_CORN8]][AMR_CHILD6], 0, 0, (N2_GPU[n]) / (1 + REF_2), 0, send_E2_corn6, receive_E2_corn6_1, receive_E2_corn6_12, E,
						&(Bufferp[n]), &(BuffersendE2corn6[block[block[n][AMR_CORN8]][AMR_CHILD6]]), &(BufferrecE2corn6_1[n]), &(BufferrecE2corn6_12[n]), &(boundevent[block[block[n][AMR_CORN8]][AMR_CHILD6]][306]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN8D_2] == block[block[n][AMR_CORN8]][AMR_CHILD8] || block[n][AMR_CORN8D_2] == -100)){
				if (block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][356], &Statbound[n][356]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE2corn6_3[n], CL_FALSE, (int)0 * sizeof(double), (N2_GPU[n] + N2G) / (1 + REF_2)*sizeof(double), receive_E2_corn6_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN8]][AMR_CHILD8], 0, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, receive_E2_corn6_2, tempreceive_E2_corn6_2, receive_E2_corn6_22, E,
						&(Bufferp[n]), &(BufferrecE2corn6_3[n]), &(tempBufferrecE2corn6_3[n]), &(BufferrecE2corn6_32[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN8]][AMR_CHILD8], block[block[n][AMR_CORN8]][AMR_CHILD8], 0, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, send_E2_corn6, receive_E2_corn6_2, receive_E2_corn6_22, E,
						&(Bufferp[n]), &(BuffersendE2corn6[block[block[n][AMR_CORN8]][AMR_CHILD8]]), &(BufferrecE2corn6_3[n]), &(BufferrecE2corn6_32[n]), &(boundevent[block[block[n][AMR_CORN8]][AMR_CHILD8]][306]), calc_corr);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E3_receive_corn(double(*restrict E[NB])[NDIM], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	int n_print = block[AMR_coord_linear(0, 0, 0, 0)][AMR_CORN2D_1];

	//positive X1
	if (block[n][AMR_CORN1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == block[n][AMR_CORN1] || (block[n][AMR_CORN1D] == -100 && block[block[n][AMR_CORN1]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][453], &Statbound[n][453]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn3[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + D3) *sizeof(double), receive_E3_corn3[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN1], N1_GPU[n], 0, 0, N3_GPU[n], receive_E3_corn3, tempreceive_E3_corn3, NULL, E,
					&(Bufferp[n]), &(BufferrecE3corn3[n]), &(tempBufferrecE3corn3[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN1], block[n][AMR_CORN1], N1_GPU[n], 0, 0, N3_GPU[n], send_E3_corn3, receive_E3_corn3, NULL, E,
					&(Bufferp[n]), &(BuffersendE3corn3[block[n][AMR_CORN1]]), &(BufferrecE3corn3[n]), NULL, &(boundevent[block[n][AMR_CORN1]][453]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1){
			if (block[n][AMR_CORN1D_1] == block[block[n][AMR_CORN1]][AMR_CHILD3] || block[n][AMR_CORN1D_1] == -100){
				if (block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][303], &Statbound[n][303]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn3_5[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn3_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN1]][AMR_CHILD3], N1_GPU[n], 0, 0, (N3_GPU[n]) / (1 + REF_3), receive_E3_corn3_1, tempreceive_E3_corn3_1, receive_E3_corn3_12, E,
						&(Bufferp[n]), &(BufferrecE3corn3_5[n]), &(tempBufferrecE3corn3_5[n]), &(BufferrecE3corn3_52[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN1]][AMR_CHILD3], block[block[n][AMR_CORN1]][AMR_CHILD3], N1_GPU[n], 0, 0, (N3_GPU[n]) / (1 + REF_3), send_E3_corn3, receive_E3_corn3_1, receive_E3_corn3_12, E,
						&(Bufferp[n]), &(BuffersendE3corn3[block[block[n][AMR_CORN1]][AMR_CHILD3]]), &(BufferrecE3corn3_5[n]), &(BufferrecE3corn3_52[n]), &(boundevent[block[block[n][AMR_CORN1]][AMR_CHILD3]][303]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN1D_2] == block[block[n][AMR_CORN1]][AMR_CHILD4] || block[n][AMR_CORN1D_2] == -100)){
				if (block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][353], &Statbound[n][353]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn3_6[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn3_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN1]][AMR_CHILD4], N1_GPU[n], 0, N3_GPU[n] / (1 + REF_3), N3_GPU[n], receive_E3_corn3_2, tempreceive_E3_corn3_2, receive_E3_corn3_22, E,
						&(Bufferp[n]), &(BufferrecE3corn3_6[n]), &(tempBufferrecE3corn3_6[n]), &(BufferrecE3corn3_62[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN1]][AMR_CHILD4], block[block[n][AMR_CORN1]][AMR_CHILD4], N1_GPU[n], 0, N3_GPU[n] / (1 + REF_3), N3_GPU[n], send_E3_corn3, receive_E3_corn3_2, receive_E3_corn3_22, E,
						&(Bufferp[n]), &(BuffersendE3corn3[block[block[n][AMR_CORN1]][AMR_CHILD4]]), &(BufferrecE3corn3_6[n]), &(BufferrecE3corn3_62[n]), &(boundevent[block[block[n][AMR_CORN1]][AMR_CHILD4]][303]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN2] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == block[n][AMR_CORN2] || (block[n][AMR_CORN2D] == -100 && block[block[n][AMR_CORN2]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid

			if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][454], &Statbound[n][454]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn4[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + D3) *sizeof(double), receive_E3_corn4[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN2], N1_GPU[n], N2_GPU[n], 0, N3_GPU[n], receive_E3_corn4, tempreceive_E3_corn4, NULL, E,
					&(Bufferp[n]), &(BufferrecE3corn4[n]), &(tempBufferrecE3corn4[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN2], block[n][AMR_CORN2], N1_GPU[n], N2_GPU[n], 0, N3_GPU[n], send_E3_corn4, receive_E3_corn4, NULL, E,
					&(Bufferp[n]), &(BuffersendE3corn4[block[n][AMR_CORN2]]), &(BufferrecE3corn4[n]), NULL, &(boundevent[block[n][AMR_CORN2]][454]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN2D_1] == block[block[n][AMR_CORN2]][AMR_CHILD1] || block[n][AMR_CORN2D_1] == -100){
				if (block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][304], &Statbound[n][304]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn4_7[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn4_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN2]][AMR_CHILD1], N1_GPU[n], N2_GPU[n], 0, (N3_GPU[n]) / (1 + REF_3), receive_E3_corn4_1, tempreceive_E3_corn4_1, receive_E3_corn4_12, E,
						&(Bufferp[n]), &(BufferrecE3corn4_7[n]), &(tempBufferrecE3corn4_7[n]), &(BufferrecE3corn4_72[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN2]][AMR_CHILD1], block[block[n][AMR_CORN2]][AMR_CHILD1], N1_GPU[n], N2_GPU[n], 0, (N3_GPU[n]) / (1 + REF_3), send_E3_corn4, receive_E3_corn4_1, receive_E3_corn4_12, E,
						&(Bufferp[n]), &(BuffersendE3corn4[block[block[n][AMR_CORN2]][AMR_CHILD1]]), &(BufferrecE3corn4_7[n]), &(BufferrecE3corn4_72[n]), &(boundevent[block[block[n][AMR_CORN2]][AMR_CHILD1]][304]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN2D_2] == block[block[n][AMR_CORN2]][AMR_CHILD2] || block[n][AMR_CORN2D_2] == -100)){
				if (block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][354], &Statbound[n][354]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn4_8[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn4_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN2]][AMR_CHILD2], N1_GPU[n], N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n], receive_E3_corn4_2, tempreceive_E3_corn4_2, receive_E3_corn4_22, E,
						&(Bufferp[n]), &(BufferrecE3corn4_8[n]), &(tempBufferrecE3corn4_8[n]), &(BufferrecE3corn4_82[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN2]][AMR_CHILD2], block[block[n][AMR_CORN2]][AMR_CHILD2], N1_GPU[n], N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n], send_E3_corn4, receive_E3_corn4_2, receive_E3_corn4_22, E,
						&(Bufferp[n]), &(BuffersendE3corn4[block[block[n][AMR_CORN2]][AMR_CHILD2]]), &(BufferrecE3corn4_8[n]), &(BufferrecE3corn4_82[n]), &(boundevent[block[block[n][AMR_CORN2]][AMR_CHILD2]][304]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == block[n][AMR_CORN3] || (block[n][AMR_CORN3D] == -100 && block[block[n][AMR_CORN3]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid

			if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][451], &Statbound[n][451]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn1[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + D3) *sizeof(double), receive_E3_corn1[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN3], 0, N2_GPU[n], 0, N3_GPU[n], receive_E3_corn1, tempreceive_E3_corn1, NULL, E,
					&(Bufferp[n]), &(BufferrecE3corn1[n]), &(tempBufferrecE3corn1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN3], block[n][AMR_CORN3], 0, N2_GPU[n], 0, N3_GPU[n], send_E3_corn1, receive_E3_corn1, NULL, E,
					&(Bufferp[n]), &(BuffersendE3corn1[block[n][AMR_CORN3]]), &(BufferrecE3corn1[n]), NULL, &(boundevent[block[n][AMR_CORN3]][451]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN3D_1] == block[block[n][AMR_CORN3]][AMR_CHILD5] || block[n][AMR_CORN3D_1] == -100){
				if (block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][301], &Statbound[n][301]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn1_3[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn1_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN3]][AMR_CHILD5], 0, N2_GPU[n], 0, (N3_GPU[n]) / (1 + REF_3), receive_E3_corn1_1, tempreceive_E3_corn1_1, receive_E3_corn1_12, E,
						&(Bufferp[n]), &(BufferrecE3corn1_3[n]), &(tempBufferrecE3corn1_3[n]), &(BufferrecE3corn1_32[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN3]][AMR_CHILD5], block[block[n][AMR_CORN3]][AMR_CHILD5], 0, N2_GPU[n], 0, (N3_GPU[n]) / (1 + REF_3), send_E3_corn1, receive_E3_corn1_1, receive_E3_corn1_12, E,
						&(Bufferp[n]), &(BuffersendE3corn1[block[block[n][AMR_CORN3]][AMR_CHILD5]]), &(BufferrecE3corn1_3[n]), &(BufferrecE3corn1_32[n]), &(boundevent[block[block[n][AMR_CORN3]][AMR_CHILD5]][301]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN3D_2] == block[block[n][AMR_CORN3]][AMR_CHILD6] || block[n][AMR_CORN3D_2] == -100)){
				if (block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][351], &Statbound[n][351]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn1_4[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn1_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN3]][AMR_CHILD6], 0, N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n], receive_E3_corn1_2, tempreceive_E3_corn1_2, receive_E3_corn1_22, E,
						&(Bufferp[n]), &(BufferrecE3corn1_4[n]), &(tempBufferrecE3corn1_4[n]), &(BufferrecE3corn1_42[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN3]][AMR_CHILD6], block[block[n][AMR_CORN3]][AMR_CHILD6], 0, N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n], send_E3_corn1, receive_E3_corn1_2, receive_E3_corn1_22, E,
						&(Bufferp[n]), &(BuffersendE3corn1[block[block[n][AMR_CORN3]][AMR_CHILD6]]), &(BufferrecE3corn1_4[n]), &(BufferrecE3corn1_42[n]), &(boundevent[block[block[n][AMR_CORN3]][AMR_CHILD6]][301]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN4] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == block[n][AMR_CORN4] || (block[n][AMR_CORN4D] == -100 && block[block[n][AMR_CORN4]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][452], &Statbound[n][452]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn2[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + D3) *sizeof(double), receive_E3_corn2[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN4], 0, 0, 0, N3_GPU[n], receive_E3_corn2, tempreceive_E3_corn2, NULL, E,
					&(Bufferp[n]), &(BufferrecE3corn2[n]), &(tempBufferrecE3corn2[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN4], block[n][AMR_CORN4], 0, 0, 0, N3_GPU[n], send_E3_corn2, receive_E3_corn2, NULL, E,
					&(Bufferp[n]), &(BuffersendE3corn2[block[n][AMR_CORN4]]), &(BufferrecE3corn2[n]), NULL, &(boundevent[block[n][AMR_CORN4]][452]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN4D_1] == block[block[n][AMR_CORN4]][AMR_CHILD7] || block[n][AMR_CORN4D_1] == -100){
				if (block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][302], &Statbound[n][302]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn2_1[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn2_1[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN4]][AMR_CHILD7], 0, 0, 0, N3_GPU[n] / (1 + REF_3), receive_E3_corn2_1, tempreceive_E3_corn2_1, receive_E3_corn2_12, E,
						&(Bufferp[n]), &(BufferrecE3corn2_1[n]), &(tempBufferrecE3corn2_1[n]), &(BufferrecE3corn2_12[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN4]][AMR_CHILD7], block[block[n][AMR_CORN4]][AMR_CHILD7], 0, 0, 0, N3_GPU[n] / (1 + REF_3), send_E3_corn2, receive_E3_corn2_1, receive_E3_corn2_12, E,
						&(Bufferp[n]), &(BuffersendE3corn2[block[block[n][AMR_CORN4]][AMR_CHILD7]]), &(BufferrecE3corn2_1[n]), &(BufferrecE3corn2_12[n]), &(boundevent[block[block[n][AMR_CORN4]][AMR_CHILD7]][302]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN4D_2] == block[block[n][AMR_CORN4]][AMR_CHILD8] || block[n][AMR_CORN4D_2] == -100)){
				if (block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][352], &Statbound[n][352]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], BufferrecE3corn2_2[n], CL_FALSE, (int)0 * sizeof(double), (N3_GPU[n] + N3G) / (1 + REF_3)*sizeof(double), receive_E3_corn2_2[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN4]][AMR_CHILD8], 0, 0, N3_GPU[n] / (1 + REF_3), N3_GPU[n], receive_E3_corn2_2, tempreceive_E3_corn2_2, receive_E3_corn2_22, E,
						&(Bufferp[n]), &(BufferrecE3corn2_2[n]), &(tempBufferrecE3corn2_2[n]), &(BufferrecE3corn2_22[n]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN4]][AMR_CHILD8], block[block[n][AMR_CORN4]][AMR_CHILD8], 0, 0, N3_GPU[n] / (1 + REF_3), N3_GPU[n], send_E3_corn2, receive_E3_corn2_2, receive_E3_corn2_22, E,
						&(Bufferp[n]), &(BuffersendE3corn2[block[block[n][AMR_CORN4]][AMR_CHILD8]]), &(BufferrecE3corn2_2[n]), &(BufferrecE3corn2_22[n]), &(boundevent[block[block[n][AMR_CORN4]][AMR_CHILD8]][302]), calc_corr);
				}
			}
		}
	}
#endif
}