#include "decs_MPI.h"

/*Send boundaries of fluxes between compute nodes through MPI*/
void flux_send1(double(*restrict F1[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send1_flux(n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3, BS_2, BS_3, send2_flux, F1, &(Bufferp[nl[n]]), &(Buffersend2flux[nl[n]]),
				&(boundevent[nl[n]][120]), NULL);
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][120],0);
					rc += MPI_Isend(&Buffersend2flux[nl[n]][0], NPR*BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((120 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send2_flux[nl[n]][0], NPR*BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((120 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec4flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((140 * n_active_total + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][140]);
				}
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_5flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][145]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec4_6flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][146]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&Bufferrec4_7flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][147]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec4_8flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][148]);
			}
		}
		else{
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive4_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((140 * n_active_total + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][140]);
				}
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_5flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][145]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive4_6flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][146]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&receive4_7flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][147]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive4_8flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((140 * n_active_total + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][148]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send2_flux, F1, &(Bufferp[nl[n]]), &(Buffersend2flux[nl[n]]),
					&(boundevent[nl[n]][120]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][120],0);
						rc += MPI_Isend(&Buffersend2flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((120 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send2_flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((120 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send1_flux(n, block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2, BS_3, send4_flux, F1, &(Bufferp[nl[n]]), &(Buffersend4flux[nl[n]]),
				&(boundevent[nl[n]][140]), NULL);
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][140],0);
					rc += MPI_Isend(&Buffersend4flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((140 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send4_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((140 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec2flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((120 * n_active_total + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][120]);
				}
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_1flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][121]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_2flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][122]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_3flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][123]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec2_4flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][124]);
			}
		}
		else{
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive2_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((120 * n_active_total + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][120]);
				}
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_1flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][121]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_2flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][122]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_3flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][123]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_2 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive2_4flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_2 / (1 + REF_2)), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((120 * n_active_total + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][124]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, 1, 0, BS_2, 0, BS_3, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send4_flux, F1, &(Bufferp[nl[n]]), &(Buffersend4flux[nl[n]]),
					&(boundevent[nl[n]][140]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][140],0);
						rc += MPI_Isend(&Buffersend4flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((140 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send4_flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((140 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void flux_send2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send2_flux(n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3, BS_1, BS_3, send3_flux, F2, &(Bufferp[nl[n]]), &(Buffersend3flux[nl[n]]),
				&(boundevent[nl[n]][130]), NULL);
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][130],0);
					rc += MPI_Isend(&Buffersend3flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((130 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send3_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((130 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec1flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((110 * n_active_total + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][110]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_3flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][113]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec1_4flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][114]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&Bufferrec1_7flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][117]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec1_8flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][118]);
			}
		}
		else{
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive1_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((110 * n_active_total + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][110]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_3flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][113]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive1_4flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][114]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&receive1_7flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][117]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive1_8flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((110 * n_active_total + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][118]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3, BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send3_flux, F2, &(Bufferp[nl[n]]), &(Buffersend3flux[nl[n]]),
					&(boundevent[nl[n]][130]));
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][130],0);
						rc += MPI_Isend(&Buffersend3flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((130 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send3_flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((130 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send2_flux(n, block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3, BS_1, BS_3, send1_flux, F2, &(Bufferp[nl[n]]), &(Buffersend1flux[nl[n]]),
				&(boundevent[nl[n]][110]), NULL);
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][110],0);
					rc += MPI_Isend(&Buffersend1flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((110 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send1_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((110 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec3flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((130 * n_active_total + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][130]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_1flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][131]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_2flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][132]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_5flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][135]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&Bufferrec3_6flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][136]);
			}
		}
		else{
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive3_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((130 * n_active_total + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][130]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_1flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][131]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_2flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][132]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_5flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][135]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_3 == 1){
				rc += MPI_Irecv(&receive3_6flux[nl[n]][0], NPR*(BS_3 / (1 + REF_3))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((130 * n_active_total + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][136]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], 0, BS_1, 0, 1, 0, BS_3, BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send1_flux, F2, &(Bufferp[nl[n]]), &(Buffersend1flux[nl[n]]),
					&(boundevent[nl[n]][110]));
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][110],0);
						rc += MPI_Isend(&Buffersend1flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((110 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send1_flux[nl[n]][0], NPR*(BS_3) / (1 + REF_3)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((110 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void flux_send3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send3_flux(n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D1, BS_1, BS_2, send5_flux, F3, &(Bufferp[nl[n]]), &(Buffersend5flux[nl[n]]),
				&(boundevent[nl[n]][150]), NULL);
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][150],0);
					rc += MPI_Isend(&Buffersend5flux[nl[n]][0], NPR* BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((150 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send5_flux[nl[n]][0], NPR* BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((150 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec6flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((160 * n_active_total + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][160]);
				}
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_2flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][162]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&Bufferrec6_4flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][164]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&Bufferrec6_6flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][166]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
				rc += MPI_Irecv(&Bufferrec6_8flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][168]);
			}
		}
		else{
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive6_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((160 * n_active_total + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][160]);
				}
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_2flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][162]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&receive6_4flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][164]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&receive6_6flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][166]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
				rc += MPI_Irecv(&receive6_8flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((160 * n_active_total + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][168]);
			}	
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], 0, BS_1, 0, BS_2, BS_3, BS_3 + 1, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send5_flux, F3, &(Bufferp[nl[n]]), &(Buffersend5flux[nl[n]]),
					&(boundevent[nl[n]][150]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][150],0);
						rc += MPI_Isend(&Buffersend5flux[nl[n]][0], NPR*(BS_2) / (1 + REF_2)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((150 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send5_flux[nl[n]][0], NPR*(BS_2) / (1 + REF_2)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((150 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	  
	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send3_flux(n, block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D1, BS_1, BS_2, send6_flux, F3, &(Bufferp[nl[n]]), &(Buffersend6flux[nl[n]]),
				&(boundevent[nl[n]][160]), NULL);
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][160],0);
					rc += MPI_Isend(&Buffersend6flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((160 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send6_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((160 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec5flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((150 * n_active_total + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][150]);
				}
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_1flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][151]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&Bufferrec5_3flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][153]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&Bufferrec5_5flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][155]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
				rc += MPI_Irecv(&Bufferrec5_7flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][157]);
			}
		}
		else{
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive5_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((150 * n_active_total + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][150]);
				}
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_1flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][151]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 && REF_2 == 1){
				rc += MPI_Irecv(&receive5_3flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][153]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 && REF_1 == 1){
				rc += MPI_Irecv(&receive5_5flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][155]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 && REF_1 == 1 && REF_2 == 1){
				rc += MPI_Irecv(&receive5_7flux[nl[n]][0], NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((150 * n_active_total + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][157]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_flux_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], 0, BS_1, 0, BS_2, 0, 1, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send6_flux, F3, &(Bufferp[nl[n]]), &(Buffersend6flux[nl[n]]),
					&(boundevent[nl[n]][160]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][160],0);
						rc += MPI_Isend(&Buffersend6flux[nl[n]][0], NPR*(BS_2) / (1 + REF_2)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((160 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send6_flux[nl[n]][0], NPR*(BS_2) / (1 + REF_2)*(BS_1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((160 * n_active_total + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void flux_rec1(double(*restrict F1[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][120], &Statbound[nl[n]][120]);
				}
				unpack_receive1_flux(n, n, block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2, BS_3, receive2_flux, receive2_flux1, NULL, F1, &(Bufferp[nl[n]]), &(Bufferrec2flux[nl[n]]), &(Bufferrec2flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[n][AMR_NBR4], block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2, BS_3, send2_flux, receive2_flux1, NULL, F1,
					&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4]]]), &(Bufferrec2flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4]]][120]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][121], &Statbound[nl[n]][121]);
				}
				unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive2_1flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[nl[n]]), &(Bufferrec2_1flux[nl[n]]), &(Bufferrec2_1flux1[nl[n]]), &(Bufferrec2_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD5], block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send2_flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]]), &(Bufferrec2_1flux1[nl[n]]), &(Bufferrec2_1flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]][120]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][122], &Statbound[nl[n]][122]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive2_2flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_2flux[nl[n]]), &(Bufferrec2_2flux1[nl[n]]), &(Bufferrec2_2flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD6], block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send2_flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]]), &(Bufferrec2_2flux1[nl[n]]), &(Bufferrec2_2flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]][120]), calc_corr);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][123], &Statbound[nl[n]][123]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive2_3flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_3flux[nl[n]]), &(Bufferrec2_3flux1[nl[n]]), &(Bufferrec2_3flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD7], block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send2_flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]]), &(Bufferrec2_3flux1[nl[n]]), &(Bufferrec2_3flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]][120]), calc_corr);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][124], &Statbound[nl[n]][124]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive2_4flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_4flux[nl[n]]), &(Bufferrec2_4flux1[nl[n]]), &(Bufferrec2_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR4]][AMR_CHILD8], block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send2_flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]]), &(Bufferrec2_4flux1[nl[n]]), &(Bufferrec2_4flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]][120]), calc_corr);
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
					MPI_Wait(&boundreqs[nl[n]][140], &Statbound[nl[n]][140]);
				}
				unpack_receive1_flux(n, n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, receive4_flux, receive4_flux1, NULL, F1, &(Bufferp[nl[n]]), &(Bufferrec4flux[nl[n]]), &(Bufferrec4flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[n][AMR_NBR2], block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, send4_flux, receive4_flux1, NULL, F1,
					&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2]]]), &(Bufferrec4flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2]]][140]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][145], &Statbound[nl[n]][145]);
				}
				unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD1], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive4_5flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[nl[n]]), &(Bufferrec4_5flux[nl[n]]), &(Bufferrec4_5flux1[nl[n]]), &(Bufferrec4_5flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD1], block[block[n][AMR_NBR2]][AMR_CHILD1], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send4_flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]]), &(Bufferrec4_5flux1[nl[n]]), &(Bufferrec4_5flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]][140]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][146], &Statbound[nl[n]][146]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD2], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive4_6flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_6flux[nl[n]]), &(Bufferrec4_6flux1[nl[n]]), &(Bufferrec4_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD2], block[block[n][AMR_NBR2]][AMR_CHILD2], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send4_flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]]), &(Bufferrec4_6flux1[nl[n]]), &(Bufferrec4_6flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]][140]), calc_corr);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][147], &Statbound[nl[n]][147]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD3], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive4_7flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_7flux[nl[n]]), &(Bufferrec4_7flux1[nl[n]]), &(Bufferrec4_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD3], block[block[n][AMR_NBR2]][AMR_CHILD3], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send4_flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]]), &(Bufferrec4_7flux1[nl[n]]), &(Bufferrec4_7flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]][140]), calc_corr);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][148], &Statbound[nl[n]][148]);
					}
					unpack_receive1_flux(n, n, block[block[n][AMR_NBR2]][AMR_CHILD4], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), receive4_8flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_8flux[nl[n]]), &(Bufferrec4_8flux1[nl[n]]), &(Bufferrec4_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive1_flux(n, block[block[n][AMR_NBR2]][AMR_CHILD4], block[block[n][AMR_NBR2]][AMR_CHILD4], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), send4_flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]]), &(Bufferrec4_8flux1[nl[n]]), &(Bufferrec4_8flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]][140]), calc_corr);
				}
			}
		}
	}
#endif
}

void flux_rec2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][130], &Statbound[nl[n]][130]);
				}
				unpack_receive2_flux(n, n, block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, receive3_flux, receive3_flux1, NULL, F2, &(Bufferp[nl[n]]), &(Bufferrec3flux[nl[n]]), &(Bufferrec3flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[n][AMR_NBR1], block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, send3_flux, receive3_flux1, NULL, F2,
					&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1]]]), &(Bufferrec3flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1]]][130]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][131], &Statbound[nl[n]][131]);
				}
				unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, BS_1 / (1 + REF_1), 0, 1, 0, BS_3 / (1 + REF_3),
					BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive3_1flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[nl[n]]), &(Bufferrec3_1flux[nl[n]]), &(Bufferrec3_1flux1[nl[n]]), &(Bufferrec3_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD3], block[block[n][AMR_NBR1]][AMR_CHILD3], 0, BS_1 / (1 + REF_1), 0, 1, 0, BS_3 / (1 + REF_3),
					BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send3_flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]]), &(Bufferrec3_1flux1[nl[n]]), &(Bufferrec3_1flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]][130]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][132], &Statbound[nl[n]][132]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, BS_1 / (1 + REF_1), 0, 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive3_2flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_2flux[nl[n]]), &(Bufferrec3_2flux1[nl[n]]), &(Bufferrec3_2flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD4], block[block[n][AMR_NBR1]][AMR_CHILD4], 0, BS_1 / (1 + REF_1), 0, 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send3_flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]]), &(Bufferrec3_2flux1[nl[n]]), &(Bufferrec3_2flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]][130]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][135], &Statbound[nl[n]][135]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, 1, 0, BS_3 / (1 + REF_3),
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive3_5flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_5flux[nl[n]]), &(Bufferrec3_5flux1[nl[n]]), &(Bufferrec3_5flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD7], block[block[n][AMR_NBR1]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, 1, 0, BS_3 / (1 + REF_3),
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send3_flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]]), &(Bufferrec3_5flux1[nl[n]]), &(Bufferrec3_5flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]][130]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][136], &Statbound[nl[n]][136]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR1]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive3_6flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_6flux[nl[n]]), &(Bufferrec3_6flux1[nl[n]]), &(Bufferrec3_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR1]][AMR_CHILD8], block[block[n][AMR_NBR1]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send3_flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]]), &(Bufferrec3_6flux1[nl[n]]), &(Bufferrec3_6flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]][130]), calc_corr);
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
					MPI_Wait(&boundreqs[nl[n]][110], &Statbound[nl[n]][110]);
				}
				unpack_receive2_flux(n, n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1, BS_3, receive1_flux, receive1_flux1, NULL, F2, &(Bufferp[nl[n]]), &(Bufferrec1flux[nl[n]]), &(Bufferrec1flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[n][AMR_NBR3], block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1, BS_3, send1_flux, receive1_flux1, NULL, F2,
					&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3]]]), &(Bufferrec1flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3]]][110]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][113], &Statbound[nl[n]][113]);
				}
				unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, BS_1 / (1 + REF_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + REF_3),
					BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive1_3flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[nl[n]]), &(Bufferrec1_3flux[nl[n]]), &(Bufferrec1_3flux1[nl[n]]), &(Bufferrec1_3flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD1], block[block[n][AMR_NBR3]][AMR_CHILD1], 0, BS_1 / (1 + REF_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + REF_3),
					BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send1_flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]]), &(Bufferrec1_3flux1[nl[n]]), &(Bufferrec1_3flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]][110]), calc_corr);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][114], &Statbound[nl[n]][114]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, BS_1 / (1 + REF_1), BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive1_4flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_4flux[nl[n]]), &(Bufferrec1_4flux1[nl[n]]), &(Bufferrec1_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD2], block[block[n][AMR_NBR3]][AMR_CHILD2], 0, BS_1 / (1 + REF_1), BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send1_flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]]), &(Bufferrec1_4flux1[nl[n]]), &(Bufferrec1_4flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]][110]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][117], &Statbound[nl[n]][117]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + REF_3),
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive1_7flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_7flux[nl[n]]), &(Bufferrec1_7flux1[nl[n]]), &(Bufferrec1_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD5], block[block[n][AMR_NBR3]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + REF_3),
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send1_flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]]), &(Bufferrec1_7flux1[nl[n]]), &(Bufferrec1_7flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]][110]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][118], &Statbound[nl[n]][118]);
					}
					unpack_receive2_flux(n, n, block[block[n][AMR_NBR3]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), receive1_8flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_8flux[nl[n]]), &(Bufferrec1_8flux1[nl[n]]), &(Bufferrec1_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive2_flux(n, block[block[n][AMR_NBR3]][AMR_CHILD6], block[block[n][AMR_NBR3]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						BS_1 / (1 + REF_1), BS_3 / (1 + REF_3), send1_flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]]), &(Bufferrec1_8flux1[nl[n]]), &(Bufferrec1_8flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]][110]), calc_corr);
				}
			}
		}
	}
#endif
}
void flux_rec3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][150], &Statbound[nl[n]][150]);
				}
				unpack_receive3_flux(n, n, block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1, BS_2, receive5_flux, receive5_flux1, NULL, F3, &(Bufferp[nl[n]]), &(Bufferrec5flux[nl[n]]), &(Bufferrec5flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[n][AMR_NBR6], block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1, BS_2, send5_flux, receive5_flux1, NULL, F3,
					&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6]]]), &(Bufferrec5flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6]]][150]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][151], &Statbound[nl[n]][151]);
				}
				unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), 0, D3,
					BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive5_1flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[nl[n]]), &(Bufferrec5_1flux[nl[n]]), &(Bufferrec5_1flux1[nl[n]]), &(Bufferrec5_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD2], block[block[n][AMR_NBR6]][AMR_CHILD2], 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), 0, D3,
					BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send5_flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]]), &(Bufferrec5_1flux1[nl[n]]), &(Bufferrec5_1flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]][150]), calc_corr);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][153], &Statbound[nl[n]][153]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive5_3flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_3flux[nl[n]]), &(Bufferrec5_3flux1[nl[n]]), &(Bufferrec5_3flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD4], block[block[n][AMR_NBR6]][AMR_CHILD4], 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send5_flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]]), &(Bufferrec5_3flux1[nl[n]]), &(Bufferrec5_3flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]][150]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][155], &Statbound[nl[n]][155]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive5_5flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_5flux[nl[n]]), &(Bufferrec5_5flux1[nl[n]]), &(Bufferrec5_5flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD6], block[block[n][AMR_NBR6]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send5_flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]]), &(Bufferrec5_5flux1[nl[n]]), &(Bufferrec5_5flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]][150]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][157], &Statbound[nl[n]][157]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR6]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive5_7flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_7flux[nl[n]]), &(Bufferrec5_7flux1[nl[n]]), &(Bufferrec5_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR6]][AMR_CHILD8], block[block[n][AMR_NBR6]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, 0, D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send5_flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]]), &(Bufferrec5_7flux1[nl[n]]), &(Bufferrec5_7flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]][150]), calc_corr);
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
					MPI_Wait(&boundreqs[nl[n]][160], &Statbound[nl[n]][160]);
				}
				unpack_receive3_flux(n, n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, receive6_flux, receive6_flux1, NULL, F3, &(Bufferp[nl[n]]), &(Bufferrec6flux[nl[n]]), &(Bufferrec6flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[n][AMR_NBR5], block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, send6_flux, receive6_flux1, NULL, F3,
					&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5]]]), &(Bufferrec6flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5]]][160]), calc_corr);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][162], &Statbound[nl[n]][162]);
				}
				unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), BS_3, BS_3 + D3,
					BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive6_2flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[nl[n]]), &(Bufferrec6_2flux[nl[n]]), &(Bufferrec6_2flux1[nl[n]]), &(Bufferrec6_2flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD1], block[block[n][AMR_NBR5]][AMR_CHILD1], 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), BS_3, BS_3 + D3,
					BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send6_flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]]), &(Bufferrec6_2flux1[nl[n]]), &(Bufferrec6_2flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]][160]), calc_corr);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][164], &Statbound[nl[n]][164]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive6_4flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_4flux[nl[n]]), &(Bufferrec6_4flux1[nl[n]]), &(Bufferrec6_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD3], block[block[n][AMR_NBR5]][AMR_CHILD3], 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send6_flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]]), &(Bufferrec6_4flux1[nl[n]]), &(Bufferrec6_4flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]][160]), calc_corr);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][166], &Statbound[nl[n]][166]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive6_6flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_6flux[nl[n]]), &(Bufferrec6_6flux1[nl[n]]), &(Bufferrec6_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD5], block[block[n][AMR_NBR5]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send6_flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]]), &(Bufferrec6_6flux1[nl[n]]), &(Bufferrec6_6flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]][160]), calc_corr);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][168], &Statbound[nl[n]][168]);
					}
					unpack_receive3_flux(n, n, block[block[n][AMR_NBR5]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), receive6_8flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_8flux[nl[n]]), &(Bufferrec6_8flux1[nl[n]]), &(Bufferrec6_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive3_flux(n, block[block[n][AMR_NBR5]][AMR_CHILD7], block[block[n][AMR_NBR5]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), send6_flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]]), &(Bufferrec6_8flux1[nl[n]]), &(Bufferrec6_8flux2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]][160]), calc_corr);
				}
			}
		}
	}
#endif
}