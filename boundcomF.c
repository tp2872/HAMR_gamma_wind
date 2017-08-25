#include "decs.h"

void pack_send1_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		clSetKernelArg(kernel_packsend1flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend1flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend1flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend1flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend1flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend1flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend1flux[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsend1flux[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsend1flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend1flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend1flux[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend1flux[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (j2 - j1)*(z2 - z1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend1flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend1flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsend1: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++) for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
				PLOOP send[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
				PLOOP send[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send2_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		clSetKernelArg(kernel_packsend2flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend2flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend2flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend2flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend2flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend2flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend2flux[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsend2flux[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsend2flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend2flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend2flux[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend2flux[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1)*(z2 - z1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend2flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend2flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsend2: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				PLOOP send[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
				PLOOP send[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send3_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict prim[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1, cl_event *boundevent2){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1;
	if (gpu == 1){
		clSetKernelArg(kernel_packsend3flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsend3flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsend3flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsend3flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsend3flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsend3flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsend3flux[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsend3flux[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsend3flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsend3flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsend3flux[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsend3flux[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1)*(j2 - j1);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend3flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsend3flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsend3: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
				PLOOP send[n][NPR*(z - z1)*jsize*isize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] = factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
				PLOOP send[n][NPR*(z - z1)*jsize*isize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] += factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
			}
		}
	}
}

void pack_send_flux_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent1){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		clSetKernelArg(kernel_packsendfluxaverage1[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendfluxaverage1[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendfluxaverage1[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendfluxaverage1[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (j2 - j1) / (1 + REF_2)*(z2 - z1) / (1 + REF_3);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage1[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent1);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage1[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsendaverage1: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j += 1 + REF_2)for (z = z1; z < z2; z += (1 + REF_3)){
				PLOOP send[n][NPR*(i - i1) *zsize*jsize + NPR*(j - j1) / (1 + REF_2)*zsize + NPR*(z - z1) / (1 + REF_3) + k]
					= 0.25* factor*(F1[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
					+ F1[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (i = i1; i < i2; i++)for (j = j1; j < j2; j += 1 + REF_2) for (z = z1; z < z2; z += (1 + REF_3)){
				PLOOP send[n][NPR*(i - i1) *zsize*jsize + NPR*(j - j1) / (1 + REF_2)*zsize + NPR*(z - z1) / (1 + REF_3) + k]
					+= 0.25* factor*(F1[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F1[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]
					+ F1[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_flux_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict F2[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];

	if (gpu == 1){
		clSetKernelArg(kernel_packsendfluxaverage2[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendfluxaverage2[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendfluxaverage2[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendfluxaverage2[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1) / (1 + REF_1)*(z2 - z1) / (1 + REF_3);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage2[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage2[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsendaverage2: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1)for (z = z1; z < z2; z += 1 + REF_3){
				PLOOP send[n][NPR*(j - j1)*isize*zsize + NPR*(i - i1) / (1 + REF_1)*zsize + NPR*(z - z1) / (1 + REF_3) + k]
					= 0.25*factor*(F2[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F2[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
					F2[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F2[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (j = j1; j < j2; j++)for (i = i1; i < i2; i += 1 + REF_1)for (z = z1; z < z2; z += 1 + REF_3){
				PLOOP send[n][NPR*(j - j1)*isize*zsize + NPR*(i - i1) / (1 + REF_1)*zsize + NPR*(z - z1) / (1 + REF_3) + k]
					+= 0.25*factor*(F2[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F2[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k] +
					F2[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F2[n][index(n, i + N1_GPU_offset[n] + REF_1, j + N2_GPU_offset[n], z + REF_3 + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void pack_send_flux_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *send[NB], double(*restrict F3[NB])[NPR], cl_mem *Bufferp, cl_mem *Bufferboundsend, cl_event *boundevent){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	int first_timestep = block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 || block[n_rec][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL];

	if (gpu == 1){
		clSetKernelArg(kernel_packsendfluxaverage3[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_packsendfluxaverage3[n], 9, sizeof(cl_mem), (void *)&(Bufferboundsend[0]));
		clSetKernelArg(kernel_packsendfluxaverage3[n], 10, sizeof(cl_double), &factor);
		clSetKernelArg(kernel_packsendfluxaverage3[n], 11, sizeof(cl_int), &first_timestep);
		global_work_size_bound[n][0] = (i2 - i1) / (1 + REF_1)*(j2 - j1) / (1 + REF_2);
		if (block[n][AMR_NSTEP] % (2 * block[n_rec][AMR_TIMELEVEL]) == 2 * block[n_rec][AMR_TIMELEVEL] - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage3[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, boundevent);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_packsendfluxaverage3[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		if (status != 0) printf("Error fluxsendaverage3: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (first_timestep == 1){
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1)for (j = j1; j < j2; j += 1 + REF_2){
				PLOOP send[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1) / (1 + REF_1)*jsize + NPR*(j - j1) / (1 + REF_2) + k]
					= 0.25*factor*(F3[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F3[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
		else{
			for (z = z1; z < z2; z++)for (i = i1; i < i2; i += 1 + REF_1) for (j = j1; j < j2; j += 1 + REF_2){
				PLOOP send[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1) / (1 + REF_1)*jsize + NPR*(j - j1) / (1 + REF_2) + k]
					+= 0.25*factor*(F3[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[n][index(n, i + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] +
					F3[n][index(n, i + REF_1 + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
					+ F3[n][index(n, i + REF_1 + N1_GPU_offset[n], j + REF_2 + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
			}
		}
	}
}

void unpack_receive1_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent1, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		clSetKernelArg(kernel_unpackreceive1flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive1flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive1flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive1flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive1flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive1flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive1flux[n], 6, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_unpackreceive1flux[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_unpackreceive1flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive1flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive1flux[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive1flux[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive1flux[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive1flux[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive1flux[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive1flux[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive1flux[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive1flux[n], 17, sizeof(cl_double), &factor);

		global_work_size_bound[n][0] = (j2 - j1)*(z2 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive1flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent1 != NULL), boundevent1, NULL);
			if (boundevent1 != NULL) clReleaseEvent(boundevent1[0]);
		}
		else status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive1flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		clFlush(commandQueueGPU[n]);
		if (status != 0) printf("Error fluxrec1: %d \n", status);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= receive[n_rec][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= (temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]) / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= (temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= receive[n_rec][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp2[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp2[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						= (receive[n_rec][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - temp2[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++) for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]
						+= (receive[n_rec][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k] - temp2[n][NPR*(i - i1)*zsize*jsize + NPR*(j - j1)*zsize + NPR*(z - z1) + k]);
				}
			}
		}
	}
}

void unpack_receive2_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent1, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		clSetKernelArg(kernel_unpackreceive2flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive2flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive2flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive2flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive2flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive2flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive2flux[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_unpackreceive2flux[n], 7, sizeof(cl_int), &zsize);
		clSetKernelArg(kernel_unpackreceive2flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive2flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive2flux[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive2flux[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive2flux[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive2flux[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive2flux[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive2flux[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive2flux[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive2flux[n], 17, sizeof(cl_double), &factor);

		global_work_size_bound[n][0] = (i2 - i1)*(z2 - z1);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive2flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent1 != NULL), boundevent1, NULL);
			if (boundevent1 != NULL) clReleaseEvent(boundevent1[0]);
		}
		else{
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive2flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		}
		clFlush(commandQueueGPU[n]);
		if (status != 0) printf("Error fluxrec2: %d \n", status);

	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= receive[n_rec][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor;
				}
			}
			else if (calc_corr == 5){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= receive[n_rec][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp2[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp2[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						= (receive[n_rec][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - temp2[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++)for (j = j1; j < j2; j++)for (z = z1; z < z2; z++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++)for (i = i1; i < i2; i++)for (z = z1; z < z2; z++){
					PLOOP temp1[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]
						+= (receive[n_rec][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k] - temp2[n][NPR*(j - j1)*zsize*isize + NPR*(i - i1)*zsize + NPR*(z - z1) + k]);
				}
			}
		}
	}
}

void unpack_receive3_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int jsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	cl_mem *Bufferp, cl_mem *Bufferboundreceive, cl_mem *Buffertemp1, cl_mem *Buffertemp2, cl_event *boundevent1, int calc_corr){
	double factor = dt*(double)block[n][AMR_TIMELEVEL];
	if (gpu == 1){
		int timelevel = block[n][AMR_TIMELEVEL];
		int timelevel_rec = block[n_rec2][AMR_TIMELEVEL];
		clSetKernelArg(kernel_unpackreceive3flux[n], 0, sizeof(cl_int), &i1);
		clSetKernelArg(kernel_unpackreceive3flux[n], 1, sizeof(cl_int), &i2);
		clSetKernelArg(kernel_unpackreceive3flux[n], 2, sizeof(cl_int), &j1);
		clSetKernelArg(kernel_unpackreceive3flux[n], 3, sizeof(cl_int), &j2);
		clSetKernelArg(kernel_unpackreceive3flux[n], 4, sizeof(cl_int), &z1);
		clSetKernelArg(kernel_unpackreceive3flux[n], 5, sizeof(cl_int), &z2);
		clSetKernelArg(kernel_unpackreceive3flux[n], 6, sizeof(cl_int), &isize);
		clSetKernelArg(kernel_unpackreceive3flux[n], 7, sizeof(cl_int), &jsize);
		clSetKernelArg(kernel_unpackreceive3flux[n], 8, sizeof(cl_mem), (void *)&(Bufferp[0]));
		clSetKernelArg(kernel_unpackreceive3flux[n], 9, sizeof(cl_mem), (void *)&(Bufferboundreceive[0]));
		clSetKernelArg(kernel_unpackreceive3flux[n], 10, sizeof(cl_mem), (void *)&(Buffertemp1[0]));
		clSetKernelArg(kernel_unpackreceive3flux[n], 11, sizeof(cl_mem), (void *)&(Buffertemp2[0]));
		clSetKernelArg(kernel_unpackreceive3flux[n], 12, sizeof(cl_int), &calc_corr);
		clSetKernelArg(kernel_unpackreceive3flux[n], 13, sizeof(cl_int), &nstep);
		clSetKernelArg(kernel_unpackreceive3flux[n], 14, sizeof(cl_int), &block[n][AMR_NSTEP]);
		clSetKernelArg(kernel_unpackreceive3flux[n], 15, sizeof(cl_int), &timelevel);
		clSetKernelArg(kernel_unpackreceive3flux[n], 16, sizeof(cl_int), &timelevel_rec);
		clSetKernelArg(kernel_unpackreceive3flux[n], 17, sizeof(cl_double), &factor);

		global_work_size_bound[n][0] = (i2 - i1)*(j2 - j1);
		//clWaitForEvents(1, &boundevent[0]);
		if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive3flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 1 * (boundevent1 != NULL), boundevent1, NULL);
			if (boundevent1 != NULL) clReleaseEvent(boundevent1[0]);
		}
		else status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_unpackreceive3flux[n], 1, global_work_offset[n], global_work_size_bound[n], NULL, 0, NULL, NULL);
		if (status != 0) printf("Error fluxrec3: %d \n", status);
		clFlush(commandQueueGPU[n]);
	}
	else{
		int i, j, z, k;
		if (block[n_rec2][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= receive[n_rec][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= receive[n_rec][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] * factor;
				}
			}
		}
		else{
			if (calc_corr == 1 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp2[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}
			else if (calc_corr == 1){
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp2[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= factor*prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k];
				}
			}

			if (calc_corr == 1 && nstep % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						= (receive[n_rec][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - temp2[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]);
				}
			}
			else if (calc_corr == 2 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						+= temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 3 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n][AMR_TIMELEVEL] - 1 && (block[n][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_rec2][AMR_TIMELEVEL] - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP prim[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						-= temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] / factor; //times dt_old/dt_new to add in future code
				}
			}
			else if (calc_corr == 5 && block[n][AMR_NSTEP] % (2 * block[n_rec2][AMR_TIMELEVEL]) == 2 * block[n_rec2][AMR_TIMELEVEL] - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++)for (i = i1; i < i2; i++)for (j = j1; j < j2; j++){
					PLOOP temp1[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]
						+= (receive[n_rec][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k] - temp2[n][NPR*(z - z1)*isize*jsize + NPR*(i - i1)*jsize + NPR*(j - j1) + k]);
				}
			}
		}
	}
}

/*Send boundaries of fluxes between compute nodes through MPI*/
void flux_send1(double(*restrict F1[NB])[NPR], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send1_flux(n, block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n], N3_GPU[n], send2_flux, F1, &(Bufferp[n]), &(Buffersend2flux[n]),
				&(boundevent[n][120]), NULL);
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][120]);
					clReleaseEvent(boundevent[n][120]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend2flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N3_GPU[n] * N2_GPU[n] * sizeof(double), send2_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send2_flux[n][0], NPR*N3_GPU[n] * N2_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((120 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_flux[n][0], NPR * N3_GPU[n] * N2_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((140 * NB + block[n][AMR_NBR2]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][140]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive4_5flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((140 * NB + block[block[n][AMR_NBR2]][AMR_CHILD1]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][145]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive4_6flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((140 * NB + block[block[n][AMR_NBR2]][AMR_CHILD2]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][146]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive4_7flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((140 * NB + block[block[n][AMR_NBR2]][AMR_CHILD3]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][147]);
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive4_8flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((140 * NB + block[block[n][AMR_NBR2]][AMR_CHILD4]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][148]);
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_flux, F1, &(Bufferp[n]), &(Buffersend2flux[n]),
					&(boundevent[n][120]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][120]);
						clReleaseEvent(boundevent[n][120]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend2flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3)*N2_GPU[n] / (1 + REF_2)*sizeof(double), send2_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send2_flux[n][0], NPR*(N3_GPU[n]) / (1 + REF_3)*(N2_GPU[n]) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((120 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[56]);
					MPI_Request_free(&req[56]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send1_flux(n, block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n], N3_GPU[n], send4_flux, F1, &(Bufferp[n]), &(Buffersend4flux[n]),
				&(boundevent[n][140]), NULL);
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][140]);
					clReleaseEvent(boundevent[n][140]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend4flux[n], CL_TRUE, (int)0 * sizeof(double), NPR * N3_GPU[n] * N2_GPU[n] * sizeof(double), send4_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send4_flux[n][0], NPR * N3_GPU[n] * N2_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((140 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}

		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_flux[n][0], NPR * N3_GPU[n] * N2_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((120 * NB + block[n][AMR_NBR4]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][120]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_1flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((120 * NB + block[block[n][AMR_NBR4]][AMR_CHILD5]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][121]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_2flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((120 * NB + block[block[n][AMR_NBR4]][AMR_CHILD6]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][122]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive2_3flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((120 * NB + block[block[n][AMR_NBR4]][AMR_CHILD7]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][123]);
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive2_4flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N2_GPU[n] / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((120 * NB + block[block[n][AMR_NBR4]][AMR_CHILD8]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][124]);
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_flux, F1, &(Bufferp[n]), &(Buffersend4flux[n]),
					&(boundevent[n][140]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][140]);
						clReleaseEvent(boundevent[n][140]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend4flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3)*N2_GPU[n] / (1 + REF_2)*sizeof(double), send4_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send4_flux[n][0], NPR*(N3_GPU[n]) / (1 + REF_3)*(N2_GPU[n]) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((140 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[57]);
					MPI_Request_free(&req[57]);
				}
			}
		}
	}
#endif
}

void flux_send2(double(*restrict F2[NB])[NPR], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send2_flux(n, block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n], N1_GPU[n], N3_GPU[n], send3_flux, F2, &(Bufferp[n]), &(Buffersend3flux[n]),
				&(boundevent[n][130]), NULL);
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][130]);
					clReleaseEvent(boundevent[n][130]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend3flux[n], CL_TRUE, (int)0 * sizeof(double), NPR * N3_GPU[n] * N1_GPU[n] * sizeof(double), send3_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send3_flux[n][0], NPR * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((130 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_flux[n][0], NPR * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((110 * NB + block[n][AMR_NBR3]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][110]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive1_3flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((110 * NB + block[block[n][AMR_NBR3]][AMR_CHILD1]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][113]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive1_4flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((110 * NB + block[block[n][AMR_NBR3]][AMR_CHILD2]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][114]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
			rc += MPI_Irecv(&receive1_7flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((110 * NB + block[block[n][AMR_NBR3]][AMR_CHILD5]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][117]);
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive1_8flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((110 * NB + block[block[n][AMR_NBR3]][AMR_CHILD6]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][118]);
		}
		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n], N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_flux, F2, &(Bufferp[n]), &(Buffersend3flux[n]),
					&(boundevent[n][130]));
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][130]);
						clReleaseEvent(boundevent[n][130]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend3flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3)*N1_GPU[n] / (1 + REF_1)*sizeof(double), send3_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send3_flux[n][0], NPR*(N3_GPU[n]) / (1 + REF_3)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((130 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[34]);
					MPI_Request_free(&req[34]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send2_flux(n, block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n], N1_GPU[n], N3_GPU[n], send1_flux, F2, &(Bufferp[n]), &(Buffersend1flux[n]),
				&(boundevent[n][110]), NULL);
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][110]);
					clReleaseEvent(boundevent[n][110]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend1flux[n], CL_TRUE, (int)0 * sizeof(double), NPR * N3_GPU[n] * N1_GPU[n] * sizeof(double), send1_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send1_flux[n][0], NPR * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((110 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_flux[n][0], NPR * N3_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((130 * NB + block[n][AMR_NBR1]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][130]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_1flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((130 * NB + block[block[n][AMR_NBR1]][AMR_CHILD3]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][131]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_2flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((130 * NB + block[block[n][AMR_NBR1]][AMR_CHILD4]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][132]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive3_5flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((130 * NB + block[block[n][AMR_NBR1]][AMR_CHILD7]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][135]);
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
			rc += MPI_Irecv(&receive3_6flux[n][0], NPR*(N3_GPU[n] / (1 + REF_3))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((130 * NB + block[block[n][AMR_NBR1]][AMR_CHILD8]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][136]);
		}
		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n], N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_flux, F2, &(Bufferp[n]), &(Buffersend1flux[n]),
					&(boundevent[n][110]));
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][110]);
						clReleaseEvent(boundevent[n][110]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend1flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3)*N1_GPU[n] / (1 + REF_1)*sizeof(double), send1_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send1_flux[n][0], NPR*(N3_GPU[n]) / (1 + REF_3)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((110 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[35]);
					MPI_Request_free(&req[35]);
				}
			}
		}
	}
#endif
}

void flux_send3(double(*restrict F3[NB])[NPR], cl_mem Bufferp[NB], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send3_flux(n, block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D1, N1_GPU[n], N2_GPU[n], send5_flux, F3, &(Bufferp[n]), &(Buffersend5flux[n]),
				&(boundevent[n][150]), NULL);
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][150]);
					clReleaseEvent(boundevent[n][150]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend5flux[n], CL_TRUE, (int)0 * sizeof(double), NPR * N2_GPU[n] * N1_GPU[n] * sizeof(double), send5_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send5_flux[n][0], NPR* N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((150 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_flux[n][0], NPR * N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((160 * NB + block[n][AMR_NBR5]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][160]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive6_2flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((160 * NB + block[block[n][AMR_NBR5]][AMR_CHILD1]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][162]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive6_4flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((160 * NB + block[block[n][AMR_NBR5]][AMR_CHILD3]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][164]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
			rc += MPI_Irecv(&receive6_6flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((160 * NB + block[block[n][AMR_NBR5]][AMR_CHILD5]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][166]);
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive6_8flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((160 * NB + block[block[n][AMR_NBR5]][AMR_CHILD7]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][168]);
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + 1, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_flux, F3, &(Bufferp[n]), &(Buffersend5flux[n]),
					&(boundevent[n][150]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][150]);
						clReleaseEvent(boundevent[n][150]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend5flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2)*N1_GPU[n] / (1 + REF_1)*sizeof(double), send5_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send5_flux[n][0], NPR*(N2_GPU[n]) / (1 + REF_2)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((150 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[36]);
					MPI_Request_free(&req[36]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send3_flux(n, block[n][AMR_NBR6], 0, N1_GPU[n], 0, N2_GPU[n], 0, D1, N1_GPU[n], N2_GPU[n], send6_flux, F3, &(Bufferp[n]), &(Buffersend6flux[n]),
				&(boundevent[n][160]), NULL);
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					clWaitForEvents(1, &boundevent[n][160]);
					clReleaseEvent(boundevent[n][160]);
					clEnqueueReadBuffer(commandQueueGPU[n], Buffersend6flux[n], CL_TRUE, (int)0 * sizeof(double), NPR * N2_GPU[n] * N1_GPU[n] * sizeof(double), send6_flux[n], 0, NULL, NULL);
				}
				rc += MPI_Isend(&send6_flux[n][0], NPR * N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((160 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[0]);
				MPI_Request_free(&req[0]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_flux[n][0], NPR * N2_GPU[n] * N1_GPU[n], MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((150 * NB + block[n][AMR_NBR6]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][150]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
			rc += MPI_Irecv(&receive5_1flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((150 * NB + block[block[n][AMR_NBR6]][AMR_CHILD2]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][151]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive5_3flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((150 * NB + block[block[n][AMR_NBR6]][AMR_CHILD4]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][153]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
			rc += MPI_Irecv(&receive5_5flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((150 * NB + block[block[n][AMR_NBR6]][AMR_CHILD6]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][155]);
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
			&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
			rc += MPI_Irecv(&receive5_7flux[n][0], NPR*(N2_GPU[n] / (1 + REF_2))*(N1_GPU[n] / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((150 * NB + block[block[n][AMR_NBR6]][AMR_CHILD8]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[n][157]);
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], 0, N1_GPU[n], 0, N2_GPU[n], 0, 1, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_flux, F3, &(Bufferp[n]), &(Buffersend6flux[n]),
					&(boundevent[n][160]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						clWaitForEvents(1, &boundevent[n][160]);
						clReleaseEvent(boundevent[n][160]);
						clEnqueueReadBuffer(commandQueueGPU[n], Buffersend6flux[n], CL_TRUE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2)*N1_GPU[n] / (1 + REF_1)*sizeof(double), send6_flux[n], 0, NULL, NULL);
					}
					rc += MPI_Isend(&send6_flux[n][0], NPR*(N2_GPU[n]) / (1 + REF_2)*(N1_GPU[n]) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((160 * NB + n) % MPI_TAG_MAX), mpi_cartcomm, &req[37]);
					MPI_Request_free(&req[37]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void flux_rec1(double(*restrict F1[NB])[NPR], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][120], &Statbound[n][120]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N3_GPU[n] * N2_GPU[n] * sizeof(double), receive2_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_flux(n, n, block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n], N3_GPU[n], receive2_flux, receive2_flux1, NULL, F1, &(Bufferp[n]), &(Bufferrec2flux[n]), &(Bufferrec2flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[n][AMR_NBR4], block[n][AMR_NBR4], 0, 1, 0, N2_GPU[n], 0, N3_GPU[n], N2_GPU[n], N3_GPU[n], send2_flux, receive2_flux1, NULL, F1,
					&(Bufferp[n]), &(Buffersend2flux[block[n][AMR_NBR4]]), &(Bufferrec2flux1[n]), NULL, &(boundevent[block[n][AMR_NBR4]][120]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][121], &Statbound[n][121]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_1flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive2_1flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_1flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[n]), &(Bufferrec2_1flux[n]), &(Bufferrec2_1flux1[n]), &(Bufferrec2_1flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD5], block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[n]), &(Buffersend2flux[block[block[n][AMR_NBR4]][AMR_CHILD5]]), &(Bufferrec2_1flux1[n]), &(Bufferrec2_1flux2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD5]][120]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][122], &Statbound[n][122]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_2flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive2_2flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_2flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[n]), &(Bufferrec2_2flux[n]), &(Bufferrec2_2flux1[n]), &(Bufferrec2_2flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD6], block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[n]), &(Buffersend2flux[block[block[n][AMR_NBR4]][AMR_CHILD6]]), &(Bufferrec2_2flux1[n]), &(Bufferrec2_2flux2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD6]][120]), calc_corr);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][123], &Statbound[n][123]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_3flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive2_3flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_3flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[n]), &(Bufferrec2_3flux[n]), &(Bufferrec2_3flux1[n]), &(Bufferrec2_3flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD7], block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[n]), &(Buffersend2flux[block[block[n][AMR_NBR4]][AMR_CHILD7]]), &(Bufferrec2_3flux1[n]), &(Bufferrec2_3flux2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD7]][120]), calc_corr);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][124], &Statbound[n][124]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec2_4flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive2_4flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive2_4flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[n]), &(Bufferrec2_4flux[n]), &(Bufferrec2_4flux1[n]), &(Bufferrec2_4flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD8], block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send2_flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[n]), &(Buffersend2flux[block[block[n][AMR_NBR4]][AMR_CHILD8]]), &(Bufferrec2_4flux1[n]), &(Bufferrec2_4flux2[n]), &(boundevent[block[block[n][AMR_NBR4]][AMR_CHILD8]][120]), calc_corr);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][140], &Statbound[n][140]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N3_GPU[n] * N2_GPU[n] * sizeof(double), receive4_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_flux(n, n, block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n], receive4_flux, receive4_flux1, NULL, F1, &(Bufferp[n]), &(Bufferrec4flux[n]), &(Bufferrec4flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[n][AMR_NBR2], block[n][AMR_NBR2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n], 0, N3_GPU[n],
					N2_GPU[n], N3_GPU[n], send4_flux, receive4_flux1, NULL, F1,
					&(Bufferp[n]), &(Buffersend2flux[block[n][AMR_NBR2]]), &(Bufferrec4flux1[n]), NULL, &(boundevent[block[n][AMR_NBR2]][140]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][145], &Statbound[n][145]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_5flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive4_5flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_5flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[n]), &(Bufferrec4_5flux[n]), &(Bufferrec4_5flux1[n]), &(Bufferrec4_5flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD1], block[block[n][AMR_NBR2]][AMR_CHILD1], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), 0, N3_GPU[n] / (1 + REF_3),
					N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[n]), &(Buffersend4flux[block[block[n][AMR_NBR2]][AMR_CHILD1]]), &(Bufferrec4_5flux1[n]), &(Bufferrec4_5flux2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD1]][140]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][146], &Statbound[n][146]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_6flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive4_6flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_6flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[n]), &(Bufferrec4_6flux[n]), &(Bufferrec4_6flux1[n]), &(Bufferrec4_6flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD2], block[block[n][AMR_NBR2]][AMR_CHILD2], N1_GPU[n], N1_GPU[n] + 1, 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[n]), &(Buffersend4flux[block[block[n][AMR_NBR2]][AMR_CHILD2]]), &(Bufferrec4_6flux1[n]), &(Bufferrec4_6flux2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD2]][140]), calc_corr);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][147], &Statbound[n][147]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_7flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive4_7flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_7flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[n]), &(Bufferrec4_7flux[n]), &(Bufferrec4_7flux1[n]), &(Bufferrec4_7flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD3], block[block[n][AMR_NBR2]][AMR_CHILD3], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, N3_GPU[n] / (1 + REF_3),
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[n]), &(Buffersend4flux[block[block[n][AMR_NBR2]][AMR_CHILD3]]), &(Bufferrec4_7flux1[n]), &(Bufferrec4_7flux2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD3]][140]), calc_corr);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][148], &Statbound[n][148]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec4_8flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N2_GPU[n] / (1 + REF_2)*sizeof(double), receive4_8flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), receive4_8flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[n]), &(Bufferrec4_8flux[n]), &(Bufferrec4_8flux1[n]), &(Bufferrec4_8flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD4], block[block[n][AMR_NBR2]][AMR_CHILD4], N1_GPU[n], N1_GPU[n] + 1, N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N2_GPU[n] / (1 + REF_2), N3_GPU[n] / (1 + REF_3), send4_flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[n]), &(Buffersend4flux[block[block[n][AMR_NBR2]][AMR_CHILD4]]), &(Bufferrec4_8flux1[n]), &(Bufferrec4_8flux2[n]), &(boundevent[block[block[n][AMR_NBR2]][AMR_CHILD4]][140]), calc_corr);
				}
			}
		}
	}
#endif
}

void flux_rec2(double(*restrict F2[NB])[NPR], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][130], &Statbound[n][130]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N1_GPU[n] * N3_GPU[n] * sizeof(double), receive3_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_flux(n, n, block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], receive3_flux, receive3_flux1, NULL, F2, &(Bufferp[n]), &(Bufferrec3flux[n]), &(Bufferrec3flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[n][AMR_NBR1], block[n][AMR_NBR1], 0, N1_GPU[n], 0, 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], send3_flux, receive3_flux1, NULL, F2,
					&(Bufferp[n]), &(Buffersend3flux[block[n][AMR_NBR1]]), &(Bufferrec3flux1[n]), NULL, &(boundevent[block[n][AMR_NBR1]][130]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][131], &Statbound[n][131]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_1flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive3_1flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), 0, 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_1flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[n]), &(Bufferrec3_1flux[n]), &(Bufferrec3_1flux1[n]), &(Bufferrec3_1flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD3], block[block[n][AMR_NBR1]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), 0, 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[n]), &(Buffersend3flux[block[block[n][AMR_NBR1]][AMR_CHILD3]]), &(Bufferrec3_1flux1[n]), &(Bufferrec3_1flux2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD3]][130]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][132], &Statbound[n][132]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_2flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive3_2flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_2flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[n]), &(Bufferrec3_2flux[n]), &(Bufferrec3_2flux1[n]), &(Bufferrec3_2flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD4], block[block[n][AMR_NBR1]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[n]), &(Buffersend3flux[block[block[n][AMR_NBR1]][AMR_CHILD4]]), &(Bufferrec3_2flux1[n]), &(Bufferrec3_2flux2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD4]][130]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][135], &Statbound[n][135]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_5flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive3_5flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_5flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[n]), &(Bufferrec3_5flux[n]), &(Bufferrec3_5flux1[n]), &(Bufferrec3_5flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD7], block[block[n][AMR_NBR1]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[n]), &(Buffersend3flux[block[block[n][AMR_NBR1]][AMR_CHILD7]]), &(Bufferrec3_5flux1[n]), &(Bufferrec3_5flux2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD7]][130]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][136], &Statbound[n][136]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec3_6flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive3_6flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive3_6flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[n]), &(Bufferrec3_6flux[n]), &(Bufferrec3_6flux1[n]), &(Bufferrec3_6flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD8], block[block[n][AMR_NBR1]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send3_flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[n]), &(Buffersend3flux[block[block[n][AMR_NBR1]][AMR_CHILD8]]), &(Bufferrec3_6flux1[n]), &(Bufferrec3_6flux2[n]), &(boundevent[block[block[n][AMR_NBR1]][AMR_CHILD8]][130]), calc_corr);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][110], &Statbound[n][110]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N1_GPU[n] * N3_GPU[n] * sizeof(double), receive1_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_flux(n, n, block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], receive1_flux, receive1_flux1, NULL, F2, &(Bufferp[n]), &(Bufferrec1flux[n]), &(Bufferrec1flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[n][AMR_NBR3], block[n][AMR_NBR3], 0, N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n],
					N1_GPU[n], N3_GPU[n], send1_flux, receive1_flux1, NULL, F2,
					&(Bufferp[n]), &(Buffersend1flux[block[n][AMR_NBR3]]), &(Bufferrec1flux1[n]), NULL, &(boundevent[block[n][AMR_NBR3]][110]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][113], &Statbound[n][113]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_3flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive1_3flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_3flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[n]), &(Bufferrec1_3flux[n]), &(Bufferrec1_3flux1[n]), &(Bufferrec1_3flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD1], block[block[n][AMR_NBR3]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
					N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[n]), &(Buffersend1flux[block[block[n][AMR_NBR3]][AMR_CHILD1]]), &(Bufferrec1_3flux1[n]), &(Bufferrec1_3flux2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD1]][110]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][114], &Statbound[n][114]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_4flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive1_4flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_4flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[n]), &(Bufferrec1_4flux[n]), &(Bufferrec1_4flux1[n]), &(Bufferrec1_4flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD2], block[block[n][AMR_NBR3]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[n]), &(Buffersend1flux[block[block[n][AMR_NBR3]][AMR_CHILD2]]), &(Bufferrec1_4flux1[n]), &(Bufferrec1_4flux2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD2]][110]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][117], &Statbound[n][117]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_7flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive1_7flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_7flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[n]), &(Bufferrec1_7flux[n]), &(Bufferrec1_7flux1[n]), &(Bufferrec1_7flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD5], block[block[n][AMR_NBR3]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, 0, N3_GPU[n] / (1 + REF_3),
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[n]), &(Buffersend1flux[block[block[n][AMR_NBR3]][AMR_CHILD5]]), &(Bufferrec1_7flux1[n]), &(Bufferrec1_7flux2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD5]][110]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][118], &Statbound[n][118]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec1_8flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N3_GPU[n] / (1 + REF_3) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive1_8flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), receive1_8flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[n]), &(Bufferrec1_8flux[n]), &(Bufferrec1_8flux1[n]), &(Bufferrec1_8flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD6], block[block[n][AMR_NBR3]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n], N2_GPU[n] + 1, N3_GPU[n] / (1 + REF_3), N3_GPU[n],
						N1_GPU[n] / (1 + REF_1), N3_GPU[n] / (1 + REF_3), send1_flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[n]), &(Buffersend1flux[block[block[n][AMR_NBR3]][AMR_CHILD6]]), &(Bufferrec1_8flux1[n]), &(Bufferrec1_8flux2[n]), &(boundevent[block[block[n][AMR_NBR3]][AMR_CHILD6]][110]), calc_corr);
				}
			}
		}
	}
#endif
}
void flux_rec3(double(*restrict F3[NB])[NPR], cl_mem Bufferp[NB], int n, int calc_corr){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][150], &Statbound[n][150]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N1_GPU[n] * N2_GPU[n] * sizeof(double), receive5_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_flux(n, n, block[n][AMR_NBR6], 0, N1_GPU[n], 0, N2_GPU[n], 0, D3,
					N1_GPU[n], N2_GPU[n], receive5_flux, receive5_flux1, NULL, F3, &(Bufferp[n]), &(Bufferrec5flux[n]), &(Bufferrec5flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[n][AMR_NBR6], block[n][AMR_NBR6], 0, N1_GPU[n], 0, N2_GPU[n], 0, D3,
					N1_GPU[n], N2_GPU[n], send5_flux, receive5_flux1, NULL, F3,
					&(Bufferp[n]), &(Buffersend5flux[block[n][AMR_NBR6]]), &(Bufferrec5flux1[n]), NULL, &(boundevent[block[n][AMR_NBR6]][150]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][151], &Statbound[n][151]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_1flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive5_1flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), 0, D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_1flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[n]), &(Bufferrec5_1flux[n]), &(Bufferrec5_1flux1[n]), &(Bufferrec5_1flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD2], block[block[n][AMR_NBR6]][AMR_CHILD2], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), 0, D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[n]), &(Buffersend5flux[block[block[n][AMR_NBR6]][AMR_CHILD2]]), &(Bufferrec5_1flux1[n]), &(Bufferrec5_1flux2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD2]][150]), calc_corr);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][153], &Statbound[n][153]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_3flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive5_3flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_3flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[n]), &(Bufferrec5_3flux[n]), &(Bufferrec5_3flux1[n]), &(Bufferrec5_3flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD4], block[block[n][AMR_NBR6]][AMR_CHILD4], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[n]), &(Buffersend5flux[block[block[n][AMR_NBR6]][AMR_CHILD4]]), &(Bufferrec5_3flux1[n]), &(Bufferrec5_3flux2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD4]][150]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][155], &Statbound[n][155]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_5flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive5_5flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_5flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[n]), &(Bufferrec5_5flux[n]), &(Bufferrec5_5flux1[n]), &(Bufferrec5_5flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD6], block[block[n][AMR_NBR6]][AMR_CHILD6], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[n]), &(Buffersend5flux[block[block[n][AMR_NBR6]][AMR_CHILD6]]), &(Bufferrec5_5flux1[n]), &(Bufferrec5_5flux2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD6]][150]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][157], &Statbound[n][157]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec5_7flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive5_7flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive5_7flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[n]), &(Bufferrec5_7flux[n]), &(Bufferrec5_7flux1[n]), &(Bufferrec5_7flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD8], block[block[n][AMR_NBR6]][AMR_CHILD8], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], 0, D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send5_flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[n]), &(Buffersend5flux[block[block[n][AMR_NBR6]][AMR_CHILD8]]), &(Bufferrec5_7flux1[n]), &(Bufferrec5_7flux2[n]), &(boundevent[block[block[n][AMR_NBR6]][AMR_CHILD8]][150]), calc_corr);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][160], &Statbound[n][160]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6flux[n], CL_FALSE, (int)0 * sizeof(double), NPR * N1_GPU[n] * N2_GPU[n] * sizeof(double), receive6_flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_flux(n, n, block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n], N2_GPU[n], receive6_flux, receive6_flux1, NULL, F3, &(Bufferp[n]), &(Bufferrec6flux[n]), &(Bufferrec6flux1[n]), NULL, NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[n][AMR_NBR5], block[n][AMR_NBR5], 0, N1_GPU[n], 0, N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n], N2_GPU[n], send6_flux, receive6_flux1, NULL, F3,
					&(Bufferp[n]), &(Buffersend6flux[block[n][AMR_NBR5]]), &(Bufferrec6flux1[n]), NULL, &(boundevent[block[n][AMR_NBR5]][160]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[n][162], &Statbound[n][162]);
					if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_2flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive6_2flux[n], 0, NULL, NULL);
					if (gpu == 1) clFlush(commandQueueGPU[n]);
				}
				unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_2flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[n]), &(Bufferrec6_2flux[n]), &(Bufferrec6_2flux1[n]), &(Bufferrec6_2flux2[n]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD1], block[block[n][AMR_NBR5]][AMR_CHILD1], 0, N1_GPU[n] / (1 + REF_1), 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
					N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[n]), &(Buffersend6flux[block[block[n][AMR_NBR5]][AMR_CHILD1]]), &(Bufferrec6_2flux1[n]), &(Bufferrec6_2flux2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD1]][160]), calc_corr);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][164], &Statbound[n][164]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_4flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive6_4flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_4flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[n]), &(Bufferrec6_4flux[n]), &(Bufferrec6_4flux1[n]), &(Bufferrec6_4flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD3], block[block[n][AMR_NBR5]][AMR_CHILD3], 0, N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[n]), &(Buffersend6flux[block[block[n][AMR_NBR5]][AMR_CHILD3]]), &(Bufferrec6_4flux1[n]), &(Bufferrec6_4flux2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD3]][160]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][166], &Statbound[n][166]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_6flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive6_6flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_6flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[n]), &(Bufferrec6_6flux[n]), &(Bufferrec6_6flux1[n]), &(Bufferrec6_6flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD5], block[block[n][AMR_NBR5]][AMR_CHILD5], N1_GPU[n] / (1 + REF_1), N1_GPU[n], 0, N2_GPU[n] / (1 + REF_2), N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[n]), &(Buffersend6flux[block[block[n][AMR_NBR5]][AMR_CHILD5]]), &(Bufferrec6_6flux1[n]), &(Bufferrec6_6flux2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD5]][160]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[n][168], &Statbound[n][168]);
						if (gpu == 1) clEnqueueWriteBuffer(commandQueueGPU[n], Bufferrec6_8flux[n], CL_FALSE, (int)0 * sizeof(double), NPR*N2_GPU[n] / (1 + REF_2) *N1_GPU[n] / (1 + REF_1)*sizeof(double), receive6_8flux[n], 0, NULL, NULL);
						if (gpu == 1) clFlush(commandQueueGPU[n]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), receive6_8flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[n]), &(Bufferrec6_8flux[n]), &(Bufferrec6_8flux1[n]), &(Bufferrec6_8flux2[n]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD7], block[block[n][AMR_NBR5]][AMR_CHILD7], N1_GPU[n] / (1 + REF_1), N1_GPU[n], N2_GPU[n] / (1 + REF_2), N2_GPU[n], N3_GPU[n], N3_GPU[n] + D3,
						N1_GPU[n] / (1 + REF_1), N2_GPU[n] / (1 + REF_2), send6_flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[n]), &(Buffersend6flux[block[block[n][AMR_NBR5]][AMR_CHILD7]]), &(Bufferrec6_8flux1[n]), &(Bufferrec6_8flux2[n]), &(boundevent[block[block[n][AMR_NBR5]][AMR_CHILD7]][160]), calc_corr);
				}
			}
		}
	}
#endif
}