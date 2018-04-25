#include "decs_MPI.h"

/*Send boundaries of Ees between compute nodes through MPI*/
void E_send1(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//MPI_Barrier(mpi_cartcomm);

	//Exchange boundary cells for MPI threads
	//Positive X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send1_E(n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2 + 2 * D2, 0, BS_3 + 2 * D3, (BS_2 + 2 * D2), (BS_3 + 2 * D3), send2_E, E, Bufferp, &(Buffersend2E[nl[n]]),
				&(boundevent[nl[n]][220]));
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][220],0);
					rc += MPI_Isend(&Buffersend2E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (22 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send2_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (22 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec4E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((24 * NB_LOCAL) + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][240]);
				}
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_5E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][245]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_6E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][246]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_7E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][247]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_8E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][248]);
			}
		}
		else{
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive4_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((24 * NB_LOCAL) + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][240]);
				}
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_5E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][245]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_6E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][246]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_7E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][247]);
			}
			if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_8E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], ((24 * NB_LOCAL) + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][248]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + D1, 0, BS_2 + 2 * D2, 0, BS_3 + 2 * D3,
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send2_E, E,
					Bufferp, &(Buffersend2E[nl[n]]), &(boundevent[nl[n]][220]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][220],0);
						rc += MPI_Isend(&Buffersend2E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((22 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send2_E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], ((22 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send1_E(n, block[n][AMR_NBR4], 0, 1, 0, BS_2 + 2 * D2, 0, BS_3 + 2 * D3, (BS_2 + 2 * D2), (BS_3 + 2 * D3), send4_E, E, Bufferp, &(Buffersend4E[nl[n]]),
				&(boundevent[nl[n]][240]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][240],0);
					rc += MPI_Isend(&Buffersend4E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((24 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send4_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((24 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec2E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((22 * NB_LOCAL) + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][220]);
				}
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_1E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][221]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_2E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][222]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_3E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][223]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_4E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][224]);
			}
		}
		else{
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive2_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((22 * NB_LOCAL) + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][220]);
				}
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_1E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][221]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_2E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][222]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_3E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][223]);
			}
			if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_4E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], ((22 * NB_LOCAL) + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][224]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, 1, 0, BS_2 + 2 * D2, 0, BS_3 + 2 * D3,
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send4_E, E,
					Bufferp, &(Buffersend4E[nl[n]]), &(boundevent[nl[n]][240]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][240],0);
						rc += MPI_Isend(&Buffersend4E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((24 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send4_E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], ((24 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void E_send2(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send2_E(n, block[n][AMR_NBR3], 0, BS_1 + 2 * D1, BS_2, BS_2 + 1, 0, BS_3 + 2 * D3, (BS_1 + 2 * D1), (BS_3 + 2 * D3), send3_E, E, Bufferp, &(Buffersend3E[nl[n]]),
				&(boundevent[nl[n]][230]));
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][230],0);
					rc += MPI_Isend(&Buffersend3E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((23 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send3_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((23 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec1E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((21 * NB_LOCAL) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][210]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_3E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][213]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_4E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][214]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_7E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][217]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_8E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][218]);
			}
		}
		else{
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive1_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((21 * NB_LOCAL) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][210]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_3E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][213]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_4E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][214]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_7E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][217]);
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_8E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], ((21 * NB_LOCAL) + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][218]);
			}
		}
		if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], 0, BS_1 + 2 * D1, BS_2, BS_2 + 1, 0, BS_3 + 2 * D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send3_E, E,
					Bufferp, &(Buffersend3E[nl[n]]), &(boundevent[nl[n]][230]));
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][230],0);
						rc += MPI_Isend(&Buffersend3E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((23 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send3_E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], ((23 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send2_E(n, block[n][AMR_NBR1], 0, BS_1 + 2 * D1, 0, 1, 0, BS_3 + 2 * D3, (BS_1 + 2 * D1), (BS_3 + 2 * D3), send1_E, E, Bufferp, &(Buffersend1E[nl[n]]),
				&(boundevent[nl[n]][210]));
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][210],0);
					rc += MPI_Isend(&Buffersend1E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((21 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send1_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((21 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec3E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((23 * NB_LOCAL) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][230]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_1E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][231]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_2E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][232]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_5E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][235]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_6E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][236]);
			}
		}
		else{
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive3_E[nl[n]][0], 2 * (BS_3 + 2 * D3)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((23 * NB_LOCAL) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][230]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_1E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][231]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_2E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][232]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_5E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][235]);
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_6E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], ((23 * NB_LOCAL) + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][236]);
			}
		}
		if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], 0, BS_1 + 2 * D1, 0, 1, 0, BS_3 + 2 * D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send1_E, E,
					Bufferp, &(Buffersend1E[nl[n]]), &(boundevent[nl[n]][210]));
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][210],0);
						rc += MPI_Isend(&Buffersend1E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((21 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send1_E[nl[n]][0], 2 * (BS_3 + 2 * D3) / (1 + REF_3)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], ((21 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void E_send3(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send3_E(n, block[n][AMR_NBR5], 0, BS_1 + 2 * D1, 0, BS_2 + 2 * D2, BS_3, BS_3 + D3, (BS_1 + 2 * D1), (BS_2 + 2 * D2), send5_E, E, Bufferp, &(Buffersend5E[nl[n]]),
				&(boundevent[nl[n]][250]));
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][250],0);
					rc += MPI_Isend(&Buffersend5E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((25 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send5_E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((25 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec6E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((26 * NB_LOCAL) + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][260]);
				}
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_2E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][262]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_4E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][264]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_6E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][266]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_8E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][268]);
			}
		}
		else{
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive6_E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((26 * NB_LOCAL) + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][260]);
				}
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_2E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][262]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_4E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][264]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_6E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][266]);
			}
			if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_8E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], ((26 * NB_LOCAL) + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][268]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], 0, BS_1 + 2 * D1, 0, BS_2 + 2 * D2, BS_3, BS_3 + D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send5_E, E,
					Bufferp, &(Buffersend5E[nl[n]]), &(boundevent[nl[n]][250]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][250],0);
						rc += MPI_Isend(&Buffersend5E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((25 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send5_E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], ((25 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] >= block[n][AMR_TIMELEVEL]){
			pack_send3_E(n, block[n][AMR_NBR6], 0, BS_1 + 2 * D1, 0, BS_2 + 2 * D2, 0, D3, (BS_1 + 2 * D1), (BS_2 + 2 * D2), send6_E, E, Bufferp, &(Buffersend6E[nl[n]]),
				&(boundevent[nl[n]][260]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][260],0);
					rc += MPI_Isend(&Buffersend6E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((26 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send6_E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((26 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec5E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((25 * NB_LOCAL) + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][250]);
				}
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_1E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][251]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_3E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][253]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_5E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][255]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_7E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][257]);
			}
		}
		else{
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive5_E[nl[n]][0], 2 * (BS_2 + 2 * D2)*(BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((25 * NB_LOCAL) + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][250]);
				}
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_1E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][251]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_3E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][253]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_5E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][255]);
			}
			if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1 && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_7E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], ((25 * NB_LOCAL) + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][257]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//send to coarser grid
				pack_send_E_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], 0, BS_1 + 2 * D1, 0, BS_2 + 2 * D2, 0, D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send6_E, E,
					Bufferp, &(Buffersend6E[nl[n]]), &(boundevent[nl[n]][260]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][260],0);
						rc += MPI_Isend(&Buffersend6E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((26 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send6_E[nl[n]][0], 2 * (BS_2 + 2 * D2) / (1 + REF_2)*(BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], ((26 * NB_LOCAL) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E_rec1(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	int flag;
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE2] == 0) MPI_Test(&boundreqs[nl[n]][220], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][220], &Statbound[nl[n]][220]);
					else if (block[n][AMR_IPROBE2] != 1) block[n][AMR_IPROBE2] = -1;
				}
				if (block[n][AMR_IPROBE2] == 0) unpack_receive1_E(n, n, block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2 + 2 * D2, BS_3 + 2 * D3, receive2_E, receive2_E1, NULL, E,
					Bufferp, &(Bufferrec2E[nl[n]]), &(Bufferrec2E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR4] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR4] && block[n][AMR_CORN3D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) + 3 * (block[n][AMR_CORN3D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR4] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR4]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR4] && block[n][AMR_CORN7D] != n)* (block[block[n][AMR_NBR4]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) + 3 * (block[n][AMR_CORN7D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE2] == 0) unpack_receive1_E(n, block[n][AMR_NBR4], block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2 + 2 * D2, BS_3 + 2 * D3, send2_E, receive2_E1, NULL, E,
					Bufferp, &(Buffersend2E[nl[block[n][AMR_NBR4]]]), &(Bufferrec2E1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4]]][220]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE2_1] == 0) MPI_Test(&boundreqs[nl[n]][221], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][221], &Statbound[nl[n]][221]);
					else if (block[n][AMR_IPROBE2_1] != 1) block[n][AMR_IPROBE2_1] = -1;
				}
				if (block[n][AMR_IPROBE2_1] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive2_1E, receive2_1E1, receive2_1E2, E,
					Bufferp, &(Bufferrec2_1E[nl[n]]), &(Bufferrec2_1E1[nl[n]]), &(Bufferrec2_1E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE2_1] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD5], block[block[n][AMR_NBR4]][AMR_CHILD5], 0, 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send2_E, receive2_1E1, receive2_1E2, E,
					Bufferp, &(Buffersend2E[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]]), &(Bufferrec2_1E1[nl[n]]), &(Bufferrec2_1E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]][220]), calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR4]][AMR_CHILD5]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_2] == 0) MPI_Test(&boundreqs[nl[n]][222], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][222], &Statbound[nl[n]][222]);
						else if (block[n][AMR_IPROBE2_2] != 1) block[n][AMR_IPROBE2_2] = -1;
					}
					if (block[n][AMR_IPROBE2_2] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive2_2E, receive2_2E1, receive2_2E2, E,
						Bufferp, &(Bufferrec2_2E[nl[n]]), &(Bufferrec2_2E1[nl[n]]), &(Bufferrec2_2E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN7D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE2_2] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD6], block[block[n][AMR_NBR4]][AMR_CHILD6], 0, 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send2_E, receive2_2E1, receive2_2E2, E,
						Bufferp, &(Buffersend2E[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]]), &(Bufferrec2_2E1[nl[n]]), &(Bufferrec2_2E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]][220]), calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR4]][AMR_CHILD6]) || (block[n][AMR_CORN7D_1] == -100)));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_3] == 0) MPI_Test(&boundreqs[nl[n]][223], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][223], &Statbound[nl[n]][223]);
						else if (block[n][AMR_IPROBE2_3] != 1) block[n][AMR_IPROBE2_3] = -1;
					}
					if (block[n][AMR_IPROBE2_3] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive2_3E, receive2_3E1, receive2_3E2, E,
						Bufferp, &(Bufferrec2_3E[nl[n]]), &(Bufferrec2_3E1[nl[n]]), &(Bufferrec2_3E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN3D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE2_3] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD7], block[block[n][AMR_NBR4]][AMR_CHILD7], 0, 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send2_E, receive2_3E1, receive2_3E2, E,
						Bufferp, &(Buffersend2E[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]]), &(Bufferrec2_3E1[nl[n]]), &(Bufferrec2_3E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]][220]), calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN3D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR4]][AMR_CHILD7]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_3 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_4] == 0) MPI_Test(&boundreqs[nl[n]][224], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][224], &Statbound[nl[n]][224]);
						else if (block[n][AMR_IPROBE2_4] != 1) block[n][AMR_IPROBE2_4] = -1;
					}
					if (block[n][AMR_IPROBE2_4] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive2_4E, receive2_4E1, receive2_4E2, E,
						Bufferp, &(Bufferrec2_4E[nl[n]]), &(Bufferrec2_4E1[nl[n]]), &(Bufferrec2_4E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN3D_2] == -100)),
						block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR4]][AMR_CHILD8]) || (block[n][AMR_CORN7D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE2_4] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR4]][AMR_CHILD8], block[block[n][AMR_NBR4]][AMR_CHILD8], 0, 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send2_E, receive2_4E1, receive2_4E2, E,
						Bufferp, &(Buffersend2E[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]]), &(Bufferrec2_4E1[nl[n]]), &(Bufferrec2_4E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]][220]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE4] == 0) MPI_Test(&boundreqs[nl[n]][240], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][240], &Statbound[nl[n]][240]);
					else if (block[n][AMR_IPROBE4] != 1) block[n][AMR_IPROBE4] = -1;
				}
				if (block[n][AMR_IPROBE4] == 0) unpack_receive1_E(n, n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2 + 2 * D2, BS_3 + 2 * D3, receive4_E, receive4_E1, NULL, E, Bufferp, &(Bufferrec4E[nl[n]]), &(Bufferrec4E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR2] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) - 3 * (block[n][AMR_CORN1D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR2] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR2] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) - 3 * (block[n][AMR_CORN5D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR2]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR2] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE4] == 0) unpack_receive1_E(n, block[n][AMR_NBR2], block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2 + 2 * D2, BS_3 + 2 * D3, send4_E, receive4_E1, NULL, E,
					Bufferp, &(Buffersend4E[nl[block[n][AMR_NBR2]]]), &(Bufferrec4E1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2]]][240]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE4_1] == 0) MPI_Test(&boundreqs[nl[n]][245], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][245], &Statbound[nl[n]][245]);
					else if (block[n][AMR_IPROBE4_1] != 1) block[n][AMR_IPROBE4_1] = -1;
				}
				if (block[n][AMR_IPROBE4_1] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD1], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive4_5E, receive4_5E1, receive4_5E2, E,
					Bufferp, &(Bufferrec4_5E[nl[n]]), &(Bufferrec4_5E1[nl[n]]), &(Bufferrec4_5E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN1D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN5D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE4_1] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD1], block[block[n][AMR_NBR2]][AMR_CHILD1], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3),
					(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send4_E, receive4_5E1, receive4_5E2, E,
					Bufferp, &(Buffersend4E[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]]), &(Bufferrec4_5E1[nl[n]]), &(Bufferrec4_5E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]][240]), calc_corr,
					1 - ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN1D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR2]][AMR_CHILD1]) || (block[n][AMR_CORN5D_1] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_2] == 0) MPI_Test(&boundreqs[nl[n]][246], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][246], &Statbound[nl[n]][246]);
						else if (block[n][AMR_IPROBE4_2] != 1) block[n][AMR_IPROBE4_2] = -1;
					}
					if (block[n][AMR_IPROBE4_2] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD2], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive4_6E, receive4_6E1, receive4_6E2, E,
						Bufferp, &(Bufferrec4_6E[nl[n]]), &(Bufferrec4_6E1[nl[n]]), &(Bufferrec4_6E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN1D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN6D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE4_2] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD2], block[block[n][AMR_NBR2]][AMR_CHILD2], BS_1, BS_1 + 1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send4_E, receive4_6E1, receive4_6E2, E,
						Bufferp, &(Buffersend4E[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]]), &(Bufferrec4_6E1[nl[n]]), &(Bufferrec4_6E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]][240]), calc_corr,
						1 - ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN1D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR2]][AMR_CHILD2]) || (block[n][AMR_CORN6D_1] == -100)));
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_3] == 0) MPI_Test(&boundreqs[nl[n]][247], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][247], &Statbound[nl[n]][247]);
						else if (block[n][AMR_IPROBE4_3] != 1) block[n][AMR_IPROBE4_3] = -1;
					}
					if (block[n][AMR_IPROBE4_3] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD3], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive4_7E, receive4_7E1, receive4_7E2, E,
						Bufferp, &(Bufferrec4_7E[nl[n]]), &(Bufferrec4_7E1[nl[n]]), &(Bufferrec4_7E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN5D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE4_3] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD3], block[block[n][AMR_NBR2]][AMR_CHILD3], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3),
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send4_E, receive4_7E1, receive4_7E2, E,
						Bufferp, &(Buffersend4E[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]]), &(Bufferrec4_7E1[nl[n]]), &(Bufferrec4_7E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]][240]), calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR2]][AMR_CHILD3]) || (block[n][AMR_CORN5D_2] == -100)), block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
			if (REF_3 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_4] == 0) MPI_Test(&boundreqs[nl[n]][248], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][248], &Statbound[nl[n]][248]);
						else if (block[n][AMR_IPROBE4_4] != 1) block[n][AMR_IPROBE4_4] = -1;
					}
					if (block[n][AMR_IPROBE4_4] == 0) unpack_receive1_E(n, n, block[block[n][AMR_NBR2]][AMR_CHILD4], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), receive4_8E, receive4_8E1, receive4_8E2, E,
						Bufferp, &(Bufferrec4_8E[nl[n]]), &(Bufferrec4_8E1[nl[n]]), &(Bufferrec4_8E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN6D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE4_4] == 0) unpack_receive1_E(n, block[block[n][AMR_NBR2]][AMR_CHILD4], block[block[n][AMR_NBR2]][AMR_CHILD4], BS_1, BS_1 + 1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3,
						(BS_2 + 2 * D2) / (1 + REF_2), (BS_3 + 2 * D3) / (1 + REF_3), send4_E, receive4_8E1, receive4_8E2, E,
						Bufferp, &(Buffersend4E[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]]), &(Bufferrec4_8E1[nl[n]]), &(Bufferrec4_8E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]][240]), calc_corr,
						block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR2]][AMR_CHILD4]) || (block[n][AMR_CORN6D_2] == -100)));
				}
			}
		}
	}
#endif
}
void E_rec2(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	int flag;
	//Positive X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE3] == 0) MPI_Test(&boundreqs[nl[n]][230], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][230], &Statbound[nl[n]][230]);
					else if (block[n][AMR_IPROBE3] != 1) block[n][AMR_IPROBE3] = -1;
				}
				if (block[n][AMR_IPROBE3] == 0) unpack_receive2_E(n, n, block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1 + 2 * D1, BS_3 + 2 * D3, receive3_E, receive3_E1, NULL, E, Bufferp, &(Bufferrec3E[nl[n]]), &(Bufferrec3E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN4D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN4D] != block[n][AMR_NBR1] && block[n][AMR_CORN4D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN4D] != -100) - 3 * (block[n][AMR_CORN4D] == -100),
					-2 + (3 * (block[n][AMR_CORN1D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN1D] != block[n][AMR_NBR1] && block[n][AMR_CORN1D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN1D] != -100) + 3 * (block[n][AMR_CORN1D] == -100),
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR1] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR1]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR1] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR1]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) + 3 * (block[n][AMR_CORN9D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE3] == 0) unpack_receive2_E(n, block[n][AMR_NBR1], block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1 + 2 * D1, BS_3 + 2 * D3, send3_E, receive3_E1, NULL, E,
					Bufferp, &(Buffersend3E[nl[block[n][AMR_NBR1]]]), &(Bufferrec3E1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1]]][230]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE3_1] == 0) MPI_Test(&boundreqs[nl[n]][231], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][231], &Statbound[nl[n]][231]);
					else if (block[n][AMR_IPROBE3_1] != 1) block[n][AMR_IPROBE3_1] = -1;
				}
				if (block[n][AMR_IPROBE3_1] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), 0, 1, 0, (BS_3) / (1 + REF_3),
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive3_1E, receive3_1E1, receive3_1E2, E,
					Bufferp, &(Bufferrec3_1E[nl[n]]), &(Bufferrec3_1E1[nl[n]]), &(Bufferrec3_1E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE3_1] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD3], block[block[n][AMR_NBR1]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), 0, 1, 0, (BS_3) / (1 + REF_3),
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send3_E, receive3_1E1, receive3_1E2, E,
					Bufferp, &(Buffersend3E[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]]), &(Bufferrec3_1E1[nl[n]]), &(Bufferrec3_1E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]][230]), calc_corr,
					1 - ((block[n][AMR_CORN4D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN4D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR1]][AMR_CHILD3]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_2] == 0) MPI_Test(&boundreqs[nl[n]][232], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][232], &Statbound[nl[n]][232]);
						else if (block[n][AMR_IPROBE3_2] != 1) block[n][AMR_IPROBE3_2] = -1;
					}
					if (block[n][AMR_IPROBE3_2] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), 0, 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive3_2E, receive3_2E1, receive3_2E2, E,
						Bufferp, &(Bufferrec3_2E[nl[n]]), &(Bufferrec3_2E1[nl[n]]), &(Bufferrec3_2E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN9D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE3_2] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD4], block[block[n][AMR_NBR1]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), 0, 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send3_E, receive3_2E1, receive3_2E2, E,
						Bufferp, &(Buffersend3E[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]]), &(Bufferrec3_2E1[nl[n]]), &(Bufferrec3_2E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]][230]), calc_corr,
						1 - ((block[n][AMR_CORN4D_2] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN4D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR1]][AMR_CHILD4]) || (block[n][AMR_CORN9D_1] == -100)));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_3] == 0) MPI_Test(&boundreqs[nl[n]][235], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][235], &Statbound[nl[n]][235]);
						else if (block[n][AMR_IPROBE3_3] != 1) block[n][AMR_IPROBE3_3] = -1;
					}
					if (block[n][AMR_IPROBE3_3] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, 1, 0, (BS_3) / (1 + REF_3),
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive3_5E, receive3_5E1, receive3_5E2, E,
						Bufferp, &(Bufferrec3_5E[nl[n]]), &(Bufferrec3_5E1[nl[n]]), &(Bufferrec3_5E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN1D_1] == -100)),
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE3_3] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD7], block[block[n][AMR_NBR1]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, 1, 0, (BS_3) / (1 + REF_3),
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send3_E, receive3_5E1, receive3_5E2, E,
						Bufferp, &(Buffersend3E[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]]), &(Bufferrec3_5E1[nl[n]]), &(Bufferrec3_5E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]][230]), calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_1] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN1D_1] == -100)),
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR1]][AMR_CHILD7]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_4] == 0) MPI_Test(&boundreqs[nl[n]][236], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][236], &Statbound[nl[n]][236]);
						else if (block[n][AMR_IPROBE3_4] != 1) block[n][AMR_IPROBE3_4] = -1;
					}
					if (block[n][AMR_IPROBE3_4] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR1]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive3_6E, receive3_6E1, receive3_6E2, E,
						Bufferp, &(Bufferrec3_6E[nl[n]]), &(Bufferrec3_6E1[nl[n]]), &(Bufferrec3_6E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN1D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN1D_2] == -100)),
						block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR1]][AMR_CHILD8]) || (block[n][AMR_CORN9D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE3_4] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR1]][AMR_CHILD8], block[block[n][AMR_NBR1]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send3_E, receive3_6E1, receive3_6E2, E,
						Bufferp, &(Buffersend3E[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]]), &(Bufferrec3_6E1[nl[n]]), &(Bufferrec3_6E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]][230]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE1] == 0) MPI_Test(&boundreqs[nl[n]][210], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][210], &Statbound[nl[n]][210]);
					else if (block[n][AMR_IPROBE1] != 1) block[n][AMR_IPROBE1] = -1;
				}
				if (block[n][AMR_IPROBE1] == 0) unpack_receive2_E(n, n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1 + 2 * D1, BS_3 + 2 * D3, receive1_E, receive1_E1, NULL, E, Bufferp, &(Bufferrec1E[nl[n]]), &(Bufferrec1E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN3D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN3D] != block[n][AMR_NBR3] && block[n][AMR_CORN3D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN3D] != -100) - 3 * (block[n][AMR_CORN3D] == -100),
					-2 + (3 * (block[n][AMR_CORN2D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN2D] != block[n][AMR_NBR3] && block[n][AMR_CORN2D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN2D] != -100) + 3 * (block[n][AMR_CORN2D] == -100),
					3 - (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR3] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) - 3 * (block[n][AMR_CORN11D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR3]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR3] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE1] == 0) unpack_receive2_E(n, block[n][AMR_NBR3], block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1 + 2 * D1, BS_3 + 2 * D3, send1_E, receive1_E1, NULL, E,
					Bufferp, &(Buffersend1E[nl[block[n][AMR_NBR3]]]), &(Bufferrec1E1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3]]][210]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE1_1] == 0) MPI_Test(&boundreqs[nl[n]][213], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][213], &Statbound[nl[n]][213]);
					else if (block[n][AMR_IPROBE1_1] != 1) block[n][AMR_IPROBE1_1] = -1;
				}
				if (block[n][AMR_IPROBE1_1] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), BS_2, BS_2 + 1, 0, (BS_3) / (1 + REF_3),
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive1_3E, receive1_3E1, receive1_3E2, E,
					Bufferp, &(Bufferrec1_3E[nl[n]]), &(Bufferrec1_3E1[nl[n]]), &(Bufferrec1_3E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN3D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN11D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE1_1] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD1], block[block[n][AMR_NBR3]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), BS_2, BS_2 + 1, 0, (BS_3) / (1 + REF_3),
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send1_E, receive1_3E1, receive1_3E2, E,
					Bufferp, &(Buffersend1E[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]]), &(Bufferrec1_3E1[nl[n]]), &(Bufferrec1_3E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]][210]), calc_corr,
					1 - ((block[n][AMR_CORN3D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN3D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR3]][AMR_CHILD1]) || (block[n][AMR_CORN11D_1] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_2] == 0) MPI_Test(&boundreqs[nl[n]][214], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][214], &Statbound[nl[n]][214]);
						else if (block[n][AMR_IPROBE1_2] != 1) block[n][AMR_IPROBE1_2] = -1;
					}
					if (block[n][AMR_IPROBE1_2] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive1_4E, receive1_4E1, receive1_4E2, E,
						Bufferp, &(Bufferrec1_4E[nl[n]]), &(Bufferrec1_4E1[nl[n]]), &(Bufferrec1_4E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN3D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN10D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE1_2] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD2], block[block[n][AMR_NBR3]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send1_E, receive1_4E1, receive1_4E2, E,
						Bufferp, &(Buffersend1E[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]]), &(Bufferrec1_4E1[nl[n]]), &(Bufferrec1_4E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]][210]), calc_corr,
						1 - ((block[n][AMR_CORN3D_2] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN3D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR3]][AMR_CHILD2]) || (block[n][AMR_CORN10D_1] == -100)));
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_3] == 0) MPI_Test(&boundreqs[nl[n]][217], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][217], &Statbound[nl[n]][217]);
						else if (block[n][AMR_IPROBE1_3] != 1) block[n][AMR_IPROBE1_3] = -1;
					}
					if (block[n][AMR_IPROBE1_3] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, 0, (BS_3) / (1 + REF_3),
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive1_7E, receive1_7E1, receive1_7E2, E,
						Bufferp, &(Bufferrec1_7E[nl[n]]), &(Bufferrec1_7E1[nl[n]]), &(Bufferrec1_7E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN11D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE1_3] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD5], block[block[n][AMR_NBR3]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, 0, (BS_3) / (1 + REF_3),
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send1_E, receive1_7E1, receive1_7E2, E,
						Bufferp, &(Buffersend1E[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]]), &(Bufferrec1_7E1[nl[n]]), &(Bufferrec1_7E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]][210]), calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_1] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN2D_1] == -100)),
						1 - ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR3]][AMR_CHILD5]) || (block[n][AMR_CORN11D_2] == -100)), block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_4] == 0) MPI_Test(&boundreqs[nl[n]][218], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][218], &Statbound[nl[n]][218]);
						else if (block[n][AMR_IPROBE1_4] != 1) block[n][AMR_IPROBE1_4] = -1;
					}
					if (block[n][AMR_IPROBE1_4] == 0) unpack_receive2_E(n, n, block[block[n][AMR_NBR3]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), receive1_8E, receive1_8E1, receive1_8E2, E,
						Bufferp, &(Bufferrec1_8E[nl[n]]), &(Bufferrec1_8E1[nl[n]]), &(Bufferrec1_8E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN10D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE1_4] == 0) unpack_receive2_E(n, block[block[n][AMR_NBR3]][AMR_CHILD6], block[block[n][AMR_NBR3]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + REF_3), BS_3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_3 + 2 * D3) / (1 + REF_3), send1_E, receive1_8E1, receive1_8E2, E,
						Bufferp, &(Buffersend1E[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]]), &(Bufferrec1_8E1[nl[n]]), &(Bufferrec1_8E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]][210]), calc_corr,
						block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN2D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN2D_2] == -100)),
						block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR3]][AMR_CHILD6]) || (block[n][AMR_CORN10D_2] == -100)));
				}
			}
		}
	}
#endif
}
void E_rec3(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	int flag;
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] <= block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE5] == 0) MPI_Test(&boundreqs[nl[n]][250], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][250], &Statbound[nl[n]][250]);
					else if (block[n][AMR_IPROBE5] != 1) block[n][AMR_IPROBE5] = -1;
				}
				if (block[n][AMR_IPROBE5] == 0) unpack_receive3_E(n, n, block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1 + 2 * D1, BS_2 + 2 * D2, receive5_E, receive5_E1, NULL, E, Bufferp, &(Bufferrec5E[nl[n]]), &(Bufferrec5E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN12D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN12D] != block[n][AMR_NBR6] && block[n][AMR_CORN12D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN12D] != -100) - 3 * (block[n][AMR_CORN12D] == -100),
					-2 + (3 * (block[n][AMR_CORN11D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN11D] != block[n][AMR_NBR6] && block[n][AMR_CORN11D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN11D] != -100) + 3 * (block[n][AMR_CORN11D] == -100),
					3 - (3 * (block[n][AMR_CORN8D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN8D] != block[n][AMR_NBR6] && block[n][AMR_CORN8D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN8D] != -100) - 3 * (block[n][AMR_CORN8D] == -100),
					-2 + (3 * (block[n][AMR_CORN5D] == block[n][AMR_NBR6]) + 2 * (block[n][AMR_CORN5D] != block[n][AMR_NBR6] && block[n][AMR_CORN5D] != n)*(block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN5D] != -100) + 3 * (block[n][AMR_CORN5D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE5] == 0) unpack_receive3_E(n, block[n][AMR_NBR6], block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1 + 2 * D1, BS_2 + 2 * D2, send5_E, receive5_E1, NULL, E,
					Bufferp, &(Buffersend5E[nl[block[n][AMR_NBR6]]]), &(Bufferrec5E1[nl[n]]), &(Bufferrec5E1[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6]]][250]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE5_1] == 0) MPI_Test(&boundreqs[nl[n]][251], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][251], &Statbound[nl[n]][251]);
					else if (block[n][AMR_IPROBE5_1] != 1) block[n][AMR_IPROBE5_1] = -1;
				}
				if (block[n][AMR_IPROBE5_1] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), 0, (BS_2) / (1 + REF_2), 0, D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive5_1E, receive5_1E1, receive5_1E2, E,
					Bufferp, &(Bufferrec5_1E[nl[n]]), &(Bufferrec5_1E1[nl[n]]), &(Bufferrec5_1E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE5_1] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD2], block[block[n][AMR_NBR6]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), 0, (BS_2) / (1 + REF_2), 0, D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send5_E, receive5_1E1, receive5_1E2, E,
					Bufferp, &(Buffersend5E[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]]), &(Bufferrec5_1E1[nl[n]]), &(Bufferrec5_1E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]][250]), calc_corr,
					1 - ((block[n][AMR_CORN12D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN12D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN8D_1] == block[block[n][AMR_NBR6]][AMR_CHILD2]) || (block[n][AMR_CORN8D_1] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_2] == 0) MPI_Test(&boundreqs[nl[n]][253], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][253], &Statbound[nl[n]][253]);
						else if (block[n][AMR_IPROBE5_2] != 1) block[n][AMR_IPROBE5_2] = -1;
					}
					if (block[n][AMR_IPROBE5_2] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive5_3E, receive5_3E1, receive5_3E2, E,
						Bufferp, &(Bufferrec5_3E[nl[n]]), &(Bufferrec5_3E1[nl[n]]), &(Bufferrec5_3E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN11D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE5_2] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD4], block[block[n][AMR_NBR6]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send5_E, receive5_3E1, receive5_3E2, E,
						Bufferp, &(Buffersend5E[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]]), &(Bufferrec5_3E1[nl[n]]), &(Bufferrec5_3E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]][250]), calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_1] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN11D_1] == -100)),
						1 - ((block[n][AMR_CORN8D_2] == block[block[n][AMR_NBR6]][AMR_CHILD4]) || (block[n][AMR_CORN8D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_3] == 0) MPI_Test(&boundreqs[nl[n]][255], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][255], &Statbound[nl[n]][255]);
						else if (block[n][AMR_IPROBE5_3] != 1) block[n][AMR_IPROBE5_3] = -1;
					}
					if (block[n][AMR_IPROBE5_3] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, 0, (BS_2) / (1 + REF_2), 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive5_5E, receive5_5E1, receive5_5E2, E,
						Bufferp, &(Bufferrec5_5E[nl[n]]), &(Bufferrec5_5E1[nl[n]]), &(Bufferrec5_5E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN5D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE5_3] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD6], block[block[n][AMR_NBR6]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, 0, (BS_2) / (1 + REF_2), 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send5_E, receive5_5E1, receive5_5E2, E,
						Bufferp, &(Buffersend5E[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]]), &(Bufferrec5_5E1[nl[n]]), &(Bufferrec5_5E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]][250]), calc_corr,
						1 - ((block[n][AMR_CORN12D_2] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN12D_2] == -100)), block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_1] == block[block[n][AMR_NBR6]][AMR_CHILD6]) || (block[n][AMR_CORN5D_1] == -100)));
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_4] == 0) MPI_Test(&boundreqs[nl[n]][257], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][257], &Statbound[nl[n]][257]);
						else if (block[n][AMR_IPROBE5_4] != 1) block[n][AMR_IPROBE5_4] = -1;
					}
					if (block[n][AMR_IPROBE5_4] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR6]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive5_7E, receive5_7E1, receive5_7E2, E,
						Bufferp, &(Bufferrec5_7E[nl[n]]), &(Bufferrec5_7E1[nl[n]]), &(Bufferrec5_7E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN11D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN11D_2] == -100)),
						block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL], ((block[n][AMR_CORN5D_2] == block[block[n][AMR_NBR6]][AMR_CHILD8]) || (block[n][AMR_CORN5D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE5_4] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR6]][AMR_CHILD8], block[block[n][AMR_NBR6]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, 0, D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send5_E, receive5_7E1, receive5_7E2, E,
						Bufferp, &(Buffersend5E[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]]), &(Bufferrec5_7E1[nl[n]]), &(Bufferrec5_7E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]][250]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE6] == 0) MPI_Test(&boundreqs[nl[n]][260], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][260], &Statbound[nl[n]][260]);
					else if (block[n][AMR_IPROBE6] != 1) block[n][AMR_IPROBE6] = -1;
				}
				if (block[n][AMR_IPROBE6] == 0) 	unpack_receive3_E(n, n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1 + 2 * D1, BS_2 + 2 * D2, receive6_E, receive6_E1, NULL, E, Bufferp, &(Bufferrec6E[nl[n]]), &(Bufferrec6E1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr,
					3 - (3 * (block[n][AMR_CORN9D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN9D] != block[n][AMR_NBR5] && block[n][AMR_CORN9D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN9D] != -100) - 3 * (block[n][AMR_CORN9D] == -100),
					-2 + (3 * (block[n][AMR_CORN10D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN10D] != block[n][AMR_NBR5] && block[n][AMR_CORN10D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN10D] != -100) + 3 * (block[n][AMR_CORN10D] == -100),
					3 - (3 * (block[n][AMR_CORN7D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN7D] != block[n][AMR_NBR5] && block[n][AMR_CORN7D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN7D] != -100) - 3 * (block[n][AMR_CORN7D] == -100),
					-2 + (3 * (block[n][AMR_CORN6D] == block[n][AMR_NBR5]) + 2 * (block[n][AMR_CORN6D] != block[n][AMR_NBR5] && block[n][AMR_CORN6D] != n)*(block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL])) * (block[n][AMR_CORN6D] != -100) + 3 * (block[n][AMR_CORN6D] == -100));
			}
			else{
				if (block[n][AMR_IPROBE6] == 0) unpack_receive3_E(n, block[n][AMR_NBR5], block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1 + 2 * D1, BS_2 + 2 * D2, send6_E, receive6_E1, NULL, E,
					Bufferp, &(Buffersend6E[nl[block[n][AMR_NBR5]]]), &(Bufferrec6E1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5]]][260]), calc_corr,
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
					flag = 0;
					if (block[n][AMR_IPROBE6_1] == 0) MPI_Test(&boundreqs[nl[n]][262], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][262], &Statbound[nl[n]][262]);
					else if (block[n][AMR_IPROBE6_1] != 1) block[n][AMR_IPROBE6_1] = -1;
				}
				if (block[n][AMR_IPROBE6_1] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), 0, (BS_2) / (1 + REF_2), BS_3, BS_3 + D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive6_2E, receive6_2E1, receive6_2E2, E,
					Bufferp, &(Bufferrec6_2E[nl[n]]), &(Bufferrec6_2E1[nl[n]]), &(Bufferrec6_2E2[nl[n]]), NULL, calc_corr,
					1 - ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN9D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN7D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			else{
				if (block[n][AMR_IPROBE6_1] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD1], block[block[n][AMR_NBR5]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), 0, (BS_2) / (1 + REF_2), BS_3, BS_3 + D3,
					(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send6_E, receive6_2E1, receive6_2E2, E,
					Bufferp, &(Buffersend6E[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]]), &(Bufferrec6_2E1[nl[n]]), &(Bufferrec6_2E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]][260]), calc_corr,
					1 - ((block[n][AMR_CORN9D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN9D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL],
					1 - ((block[n][AMR_CORN7D_1] == block[block[n][AMR_NBR5]][AMR_CHILD1]) || (block[n][AMR_CORN7D_1] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_2] == 0) MPI_Test(&boundreqs[nl[n]][264], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][264], &Statbound[nl[n]][264]);
						else if (block[n][AMR_IPROBE6_2] != 1) block[n][AMR_IPROBE6_2] = -1;
					}
					if (block[n][AMR_IPROBE6_2] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive6_4E, receive6_4E1, receive6_4E2, E,
						Bufferp, &(Bufferrec6_4E[nl[n]]), &(Bufferrec6_4E1[nl[n]]), &(Bufferrec6_4E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN10D_1] == -100)),
						1 - ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN7D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					if (block[n][AMR_IPROBE6_2] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD3], block[block[n][AMR_NBR5]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send6_E, receive6_4E1, receive6_4E2, E,
						Bufferp, &(Buffersend6E[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]]), &(Bufferrec6_4E1[nl[n]]), &(Bufferrec6_4E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]][260]), calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_1] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN10D_1] == -100)),
						1 - ((block[n][AMR_CORN7D_2] == block[block[n][AMR_NBR5]][AMR_CHILD3]) || (block[n][AMR_CORN7D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_3] == 0) MPI_Test(&boundreqs[nl[n]][266], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][266], &Statbound[nl[n]][266]);
						else if (block[n][AMR_IPROBE6_3] != 1) block[n][AMR_IPROBE6_3] = -1;
					}
					if (block[n][AMR_IPROBE6_3] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, 0, (BS_2) / (1 + REF_2), BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive6_6E, receive6_6E1, receive6_6E2, E,
						Bufferp, &(Bufferrec6_6E[nl[n]]), &(Bufferrec6_6E1[nl[n]]), &(Bufferrec6_6E2[nl[n]]), NULL, calc_corr,
						1 - ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN9D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN6D_1] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE6_3] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD5], block[block[n][AMR_NBR5]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, 0, (BS_2) / (1 + REF_2), BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send6_E, receive6_6E1, receive6_6E2, E,
						Bufferp, &(Buffersend6E[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]]), &(Bufferrec6_6E1[nl[n]]), &(Bufferrec6_6E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]][260]), calc_corr,
						1 - ((block[n][AMR_CORN9D_2] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN9D_2] == -100)), block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL],
						block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_1] == block[block[n][AMR_NBR5]][AMR_CHILD5]) || (block[n][AMR_CORN6D_1] == -100)));
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_4] == 0) MPI_Test(&boundreqs[nl[n]][268], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][268], &Statbound[nl[n]][268]);
						else if (block[n][AMR_IPROBE6_4] != 1) block[n][AMR_IPROBE6_4] = -1;
					}
					if (block[n][AMR_IPROBE6_4] == 0) unpack_receive3_E(n, n, block[block[n][AMR_NBR5]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), receive6_8E, receive6_8E1, receive6_8E2, E,
						Bufferp, &(Bufferrec6_8E[nl[n]]), &(Bufferrec6_8E1[nl[n]]), &(Bufferrec6_8E2[nl[n]]), NULL, calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN10D_2] == -100)),
						block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN6D_2] == -100)));
				}
				else{
					if (block[n][AMR_IPROBE6_4] == 0) unpack_receive3_E(n, block[block[n][AMR_NBR5]][AMR_CHILD7], block[block[n][AMR_NBR5]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, BS_3 + D3,
						(BS_1 + 2 * D1) / (1 + REF_1), (BS_2 + 2 * D2) / (1 + REF_2), send6_E, receive6_8E1, receive6_8E2, E,
						Bufferp, &(Buffersend6E[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]]), &(Bufferrec6_8E1[nl[n]]), &(Bufferrec6_8E2[nl[n]]), &(boundevent[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]][260]), calc_corr,
						block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN10D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN10D_2] == -100)),
						block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] <= block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL], ((block[n][AMR_CORN6D_2] == block[block[n][AMR_NBR5]][AMR_CHILD7]) || (block[n][AMR_CORN6D_2] == -100)));
				}
			}
		}
	}
#endif
}

void E1_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
	if (block[n][AMR_CORN9] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == n || (block[n][AMR_CORN9D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN9]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN9], 0, BS_1 + D1, 0, BS_3, send_E1_corn9, E, Bufferp, &(BuffersendE1corn9[nl[n]]),
				&(boundevent[nl[n]][459]));
			if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][459],0);
					rc += MPI_Isend(&BuffersendE1corn9[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E1_corn9[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == block[n][AMR_CORN9] || (block[n][AMR_CORN9D] == -100 && block[block[n][AMR_CORN9]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE1corn11[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN9]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][461]);
				}
			}
			if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_1] == block[block[n][AMR_CORN9]][AMR_CHILD3] || block[n][AMR_CORN9D_1] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn11_2[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][311]);
			}
			if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_2] == block[block[n][AMR_CORN9]][AMR_CHILD7] || block[n][AMR_CORN9D_2] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn11_6[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][361]);
			}
		}
		else{
			if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == block[n][AMR_CORN9] || (block[n][AMR_CORN9D] == -100 && block[block[n][AMR_CORN9]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E1_corn11[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN9]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN9]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][461]);
				}
			}
			if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_1] == block[block[n][AMR_CORN9]][AMR_CHILD3] || block[n][AMR_CORN9D_1] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn11_1[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][311]);
			}
			if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1 && (block[n][AMR_CORN9D_2] == block[block[n][AMR_CORN9]][AMR_CHILD7] || block[n][AMR_CORN9D_2] == -100) && block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn11_2[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][361]);
			}
		}
		if (block[block[n][AMR_CORN9]][AMR_PARENT] >= 0 && (block[n][AMR_CORN9D] == n || block[n][AMR_CORN9D] == -100)){
			if (block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN9] == block[block[n][AMR_CORN9]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN9]][AMR_PARENT], 0, BS_1 + 2 * D1, 0, BS_3, send_E1_corn9, E,
					Bufferp, &(BuffersendE1corn9[nl[n]]), &(boundevent[nl[n]][309]));
				if (block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][309],0);
						rc += MPI_Isend(&BuffersendE1corn9[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E1_corn9[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN9]][AMR_PARENT]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN10] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == n || (block[n][AMR_CORN10D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN10]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN10], 0, BS_1 + D1, BS_2, BS_3, send_E1_corn10, E, Bufferp, &(BuffersendE1corn10[nl[n]]),
				&(boundevent[nl[n]][460]));
			if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][460],0);
					rc += MPI_Isend(&BuffersendE1corn10[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E1_corn10[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == block[n][AMR_CORN10] || (block[n][AMR_CORN10D] == -100 && block[block[n][AMR_CORN10]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE1corn12[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN10]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][462]);
				}
			}
			if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_1] == block[block[n][AMR_CORN10]][AMR_CHILD1] || block[n][AMR_CORN10D_1] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn12_4[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][312]);
			}
			if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_2] == block[block[n][AMR_CORN10]][AMR_CHILD5] || block[n][AMR_CORN10D_2] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn12_8[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][362]);
			}
		}
		else{
			if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == block[n][AMR_CORN10] || (block[n][AMR_CORN10D] == -100 && block[block[n][AMR_CORN10]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E1_corn12[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN10]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN10]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][462]);
				}
			}
			if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_1] == block[block[n][AMR_CORN10]][AMR_CHILD1] || block[n][AMR_CORN10D_1] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn12_1[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][312]);
			}
			if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1 && (block[n][AMR_CORN10D_2] == block[block[n][AMR_CORN10]][AMR_CHILD5] || block[n][AMR_CORN10D_2] == -100) && block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn12_2[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][362]);
			}
		}
		if (block[block[n][AMR_CORN10]][AMR_PARENT] >= 0 && (block[n][AMR_CORN10D] == n || block[n][AMR_CORN10D] == -100)){
			if (block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN10] == block[block[n][AMR_CORN10]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN10]][AMR_PARENT], 0, BS_1 + 2 * D1, BS_2, BS_3, send_E1_corn10, E,
					Bufferp, &(BuffersendE1corn10[nl[n]]), &(boundevent[nl[n]][310]));
				if (block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][310],0);
						rc += MPI_Isend(&BuffersendE1corn10[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E1_corn10[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN10]][AMR_PARENT]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN11] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == n || (block[n][AMR_CORN11D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN11]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN11], 0, BS_1 + D1, BS_2, 0, send_E1_corn11, E, Bufferp, &(BuffersendE1corn11[nl[n]]),
				&(boundevent[nl[n]][461]));
			if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][461],0);
					rc += MPI_Isend(&BuffersendE1corn11[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E1_corn11[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == block[n][AMR_CORN11] || (block[n][AMR_CORN11D] == -100 && block[block[n][AMR_CORN11]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE1corn9[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN11]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][459]);
				}
			}
			if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_1] == block[block[n][AMR_CORN11]][AMR_CHILD2] || block[n][AMR_CORN11D_1] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn9_3[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][309]);
			}
			if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_2] == block[block[n][AMR_CORN11]][AMR_CHILD6] || block[n][AMR_CORN11D_2] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn9_7[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][359]);
			}
		}
		else{
			if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == block[n][AMR_CORN11] || (block[n][AMR_CORN11D] == -100 && block[block[n][AMR_CORN11]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E1_corn9[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN11]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN11]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][459]);
				}
			}
			if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_1] == block[block[n][AMR_CORN11]][AMR_CHILD2] || block[n][AMR_CORN11D_1] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn9_1[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][309]);
			}
			if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1 && (block[n][AMR_CORN11D_2] == block[block[n][AMR_CORN11]][AMR_CHILD6] || block[n][AMR_CORN11D_2] == -100) && block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn9_2[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE], (9 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][359]);
			}
		}
		if (block[block[n][AMR_CORN11]][AMR_PARENT] >= 0 && (block[n][AMR_CORN11D] == n || block[n][AMR_CORN11D] == -100)){
			if (block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN11] == block[block[n][AMR_CORN11]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN11]][AMR_PARENT], 0, BS_1 + 2 * D1, BS_2, 0, send_E1_corn11, E,
					Bufferp, &(BuffersendE1corn11[nl[n]]), &(boundevent[nl[n]][311]));
				if (block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][311],0);
						rc += MPI_Isend(&BuffersendE1corn11[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E1_corn11[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN11]][AMR_PARENT]][AMR_NODE], (11 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN12] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == n || (block[n][AMR_CORN12D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN12]][AMR_TIMELEVEL]))){
			pack_send_E1_corn(n, block[n][AMR_CORN12], 0, BS_1 + D1, 0, 0, send_E1_corn12, E, Bufferp, &(BuffersendE1corn12[nl[n]]),
				&(boundevent[nl[n]][462]));
			if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][462],0);
					rc += MPI_Isend(&BuffersendE1corn12[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E1_corn12[nl[n]][0], BS_1 + 2 * D1, MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == block[n][AMR_CORN12] || (block[n][AMR_CORN12D] == -100 && block[block[n][AMR_CORN12]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE1corn10[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN12]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][460]);
				}
			}
			if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_1] == block[block[n][AMR_CORN12]][AMR_CHILD4] || block[n][AMR_CORN12D_1] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn10_1[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][310]);
			}
			if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_2] == block[block[n][AMR_CORN12]][AMR_CHILD8] || block[n][AMR_CORN12D_2] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE1corn10_5[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][360]);
			}
		}
		else{
			if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == block[n][AMR_CORN12] || (block[n][AMR_CORN12D] == -100 && block[block[n][AMR_CORN12]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E1_corn10[nl[n]][0], (BS_1 + 2 * D1), MPI_DOUBLE, block[block[n][AMR_CORN12]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN12]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][460]);
				}
			}
			if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_1] == block[block[n][AMR_CORN12]][AMR_CHILD4] || block[n][AMR_CORN12D_1] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn10_1[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][310]);
			}
			if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && (block[n][AMR_CORN12D_2] == block[block[n][AMR_CORN12]][AMR_CHILD8] || block[n][AMR_CORN12D_2] == -100) && block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E1_corn10_2[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE], (10 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][360]);
			}
		}
		if (block[block[n][AMR_CORN12]][AMR_PARENT] >= 0 && (block[n][AMR_CORN12D] == n || block[n][AMR_CORN12D] == -100)){
			if (block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN12] == block[block[n][AMR_CORN12]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E1_corn_course(n, block[block[n][AMR_CORN12]][AMR_PARENT], 0, BS_1 + 2 * D1, 0, 0, send_E1_corn12, E,
					Bufferp, &(BuffersendE1corn12[nl[n]]), &(boundevent[nl[n]][312]));
				if (block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][312],0);
						rc += MPI_Isend(&BuffersendE1corn12[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E1_corn12[nl[n]][0], (BS_1 + 2 * D1) / (1 + REF_1), MPI_DOUBLE, block[block[block[n][AMR_CORN12]][AMR_PARENT]][AMR_NODE], (12 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
}

void E2_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
	if (block[n][AMR_CORN5] >= 0){
		if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == n || (block[n][AMR_CORN5D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN5]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN5], BS_1, 0, BS_2 + D2, 0, send_E2_corn5, E, Bufferp, &(BuffersendE2corn5[nl[n]]),
				&(boundevent[nl[n]][455]));
			if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][455],0);
					rc += MPI_Isend(&BuffersendE2corn5[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E2_corn5[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == block[n][AMR_CORN5] || (block[n][AMR_CORN5D] == -100 && block[block[n][AMR_CORN5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE2corn7[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][457]);
				}
			}
			if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_1] == block[block[n][AMR_CORN5]][AMR_CHILD2] || block[n][AMR_CORN5D_1] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn7_5[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][307]);
			}
			if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_2] == block[block[n][AMR_CORN5]][AMR_CHILD4] || block[n][AMR_CORN5D_2] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn7_7[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][357]);
			}
		}
		else{
			if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == block[n][AMR_CORN5] || (block[n][AMR_CORN5D] == -100 && block[block[n][AMR_CORN5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E2_corn7[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN5]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][457]);
				}
			}
			if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_1] == block[block[n][AMR_CORN5]][AMR_CHILD2] || block[n][AMR_CORN5D_1] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn7_1[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][307]);
			}
			if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1 && (block[n][AMR_CORN5D_2] == block[block[n][AMR_CORN5]][AMR_CHILD4] || block[n][AMR_CORN5D_2] == -100) && block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn7_2[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][357]);
			}
		}
		if (block[block[n][AMR_CORN5]][AMR_PARENT] >= 0 && (block[n][AMR_CORN5D] == n || block[n][AMR_CORN5D] == -100)){
			if (block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN5] == block[block[n][AMR_CORN5]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN5]][AMR_PARENT], BS_1, 0, BS_2 + 2 * D2, 0, send_E2_corn5, E,
					Bufferp, &(BuffersendE2corn5[nl[n]]), &(boundevent[nl[n]][305]));
				if (block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][305],0);
						rc += MPI_Isend(&BuffersendE2corn5[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E2_corn5[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN5]][AMR_PARENT]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN6] >= 0){
		if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == n || (block[n][AMR_CORN6D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN6]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN6], BS_1, 0, BS_2 + D2, BS_3, send_E2_corn6, E, Bufferp, &(BuffersendE2corn6[nl[n]]),
				&(boundevent[nl[n]][456]));
			if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][456],0);
					rc += MPI_Isend(&BuffersendE2corn6[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E2_corn6[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == block[n][AMR_CORN6] || (block[n][AMR_CORN6D] == -100 && block[block[n][AMR_CORN6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE2corn8[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][458]);
				}
			}
			if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_1] == block[block[n][AMR_CORN6]][AMR_CHILD1] || block[n][AMR_CORN6D_1] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn8_6[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][308]);
			}
			if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_2] == block[block[n][AMR_CORN6]][AMR_CHILD3] || block[n][AMR_CORN6D_2] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn8_8[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][358]);
			}
		}
		else{
			if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == block[n][AMR_CORN6] || (block[n][AMR_CORN6D] == -100 && block[block[n][AMR_CORN6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E2_corn8[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN6]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][458]);
				}
			}
			if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_1] == block[block[n][AMR_CORN6]][AMR_CHILD1] || block[n][AMR_CORN6D_1] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn8_1[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][308]);
			}
			if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1 && (block[n][AMR_CORN6D_2] == block[block[n][AMR_CORN6]][AMR_CHILD3] || block[n][AMR_CORN6D_2] == -100) && block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn8_2[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][358]);
			}
		}
		if (block[block[n][AMR_CORN6]][AMR_PARENT] >= 0 && (block[n][AMR_CORN6D] == n || block[n][AMR_CORN6D] == -100)){
			if (block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN6] == block[block[n][AMR_CORN6]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN6]][AMR_PARENT], BS_1, 0, BS_2 + 2 * D2, BS_3, send_E2_corn6, E,
					Bufferp, &(BuffersendE2corn6[nl[n]]), &(boundevent[nl[n]][306]));
				if (block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][306],0);
						rc += MPI_Isend(&BuffersendE2corn6[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E2_corn6[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN6]][AMR_PARENT]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN7] >= 0){
		if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == n || (block[n][AMR_CORN7D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN7]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN7], 0, 0, BS_2 + D2, BS_3, send_E2_corn7, E, Bufferp, &(BuffersendE2corn7[nl[n]]),
				&(boundevent[nl[n]][457]));
			if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][457],0);
					rc += MPI_Isend(&BuffersendE2corn7[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E2_corn7[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == block[n][AMR_CORN7] || (block[n][AMR_CORN7D] == -100 && block[block[n][AMR_CORN7]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE2corn5[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][455]);
				}
			}
			if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_1] == block[block[n][AMR_CORN7]][AMR_CHILD5] || block[n][AMR_CORN7D_1] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn5_2[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][305]);
			}
			if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_2] == block[block[n][AMR_CORN7]][AMR_CHILD7] || block[n][AMR_CORN7D_2] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn5_4[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][355]);
			}
		}
		else{
			if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == block[n][AMR_CORN7] || (block[n][AMR_CORN7D] == -100 && block[block[n][AMR_CORN7]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E2_corn5[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN7]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][455]);
				}
			}
			if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_1] == block[block[n][AMR_CORN7]][AMR_CHILD5] || block[n][AMR_CORN7D_1] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn5_1[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][305]);
			}
			if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1 && (block[n][AMR_CORN7D_2] == block[block[n][AMR_CORN7]][AMR_CHILD7] || block[n][AMR_CORN7D_2] == -100) && block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn5_2[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE], (5 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][355]);
			}
		}

		if (block[block[n][AMR_CORN7]][AMR_PARENT] >= 0 && (block[n][AMR_CORN7D] == n || block[n][AMR_CORN7D] == -100)){
			if (block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN7] == block[block[n][AMR_CORN7]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN7]][AMR_PARENT], 0, 0, BS_2 + 2 * D2, BS_3, send_E2_corn7, E,
					Bufferp, &(BuffersendE2corn7[nl[n]]), &(boundevent[nl[n]][307]));
				if (block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][307],0);
						rc += MPI_Isend(&BuffersendE2corn7[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E2_corn7[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN7]][AMR_PARENT]][AMR_NODE], (7 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN8] >= 0){
		if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == n || (block[n][AMR_CORN8D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN8]][AMR_TIMELEVEL]))){
			pack_send_E2_corn(n, block[n][AMR_CORN8], 0, 0, BS_2 + D2, 0, send_E2_corn8, E, Bufferp, &(BuffersendE2corn8[nl[n]]),
				&(boundevent[nl[n]][458]));
			if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][458],0);
					rc += MPI_Isend(&BuffersendE2corn8[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E2_corn8[nl[n]][0], BS_2 + 2 * D2, MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == block[n][AMR_CORN8] || (block[n][AMR_CORN8D] == -100 && block[block[n][AMR_CORN8]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE2corn6[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][456]);
				}
			}
			if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_1] == block[block[n][AMR_CORN8]][AMR_CHILD6] || block[n][AMR_CORN8D_1] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn6_1[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][306]);
			}
			if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_2] == block[block[n][AMR_CORN8]][AMR_CHILD8] || block[n][AMR_CORN8D_2] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE2corn6_3[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][356]);
			}
		}
		else{
			if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == block[n][AMR_CORN8] || (block[n][AMR_CORN8D] == -100 && block[block[n][AMR_CORN8]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E2_corn6[nl[n]][0], (BS_2 + 2 * D2), MPI_DOUBLE, block[block[n][AMR_CORN8]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][456]);
				}
			}
			if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_1] == block[block[n][AMR_CORN8]][AMR_CHILD6] || block[n][AMR_CORN8D_1] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn6_1[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][306]);
			}
			if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1 && (block[n][AMR_CORN8D_2] == block[block[n][AMR_CORN8]][AMR_CHILD8] || block[n][AMR_CORN8D_2] == -100) && block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E2_corn6_2[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE], (6 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][356]);
			}
		}
		if (block[block[n][AMR_CORN8]][AMR_PARENT] >= 0 && (block[n][AMR_CORN8D] == n || block[n][AMR_CORN8D] == -100)){
			if (block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN8] == block[block[n][AMR_CORN8]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E2_corn_course(n, block[block[n][AMR_CORN8]][AMR_PARENT], 0, 0, BS_2 + 2 * D2, 0, send_E2_corn8, E,
					Bufferp, &(BuffersendE2corn8[nl[n]]), &(boundevent[nl[n]][308]));
				if (block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][308],0);
						rc += MPI_Isend(&BuffersendE2corn8[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E2_corn8[nl[n]][0], (BS_2 + 2 * D2) / (1 + REF_2), MPI_DOUBLE, block[block[block[n][AMR_CORN8]][AMR_PARENT]][AMR_NODE], (8 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
}

void E3_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n){
	if (block[n][AMR_CORN1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == n || (block[n][AMR_CORN1D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN1]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN1], BS_1, 0, 0, BS_3 + D3, send_E3_corn1, E, Bufferp, &(BuffersendE3corn1[nl[n]]),
				&(boundevent[nl[n]][451]));
			if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][451],0);
					rc += MPI_Isend(&BuffersendE3corn1[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E3_corn1[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == block[n][AMR_CORN1] || (block[n][AMR_CORN1D] == -100 && block[block[n][AMR_CORN1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE3corn3[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][453]);
				}
			}
			if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_1] == block[block[n][AMR_CORN1]][AMR_CHILD3] || block[n][AMR_CORN1D_1] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn3_5[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][303]);
			}
			if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_2] == block[block[n][AMR_CORN1]][AMR_CHILD4] || block[n][AMR_CORN1D_2] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn3_6[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][353]);
			}
		}
		else{
			if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == block[n][AMR_CORN1] || (block[n][AMR_CORN1D] == -100 && block[block[n][AMR_CORN1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E3_corn3[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN1]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][453]);
				}
			}
			if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_1] == block[block[n][AMR_CORN1]][AMR_CHILD3] || block[n][AMR_CORN1D_1] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn3_1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][303]);
			}
			if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1 && (block[n][AMR_CORN1D_2] == block[block[n][AMR_CORN1]][AMR_CHILD4] || block[n][AMR_CORN1D_2] == -100) && block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn3_2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][353]);
			}
		}
		if (block[block[n][AMR_CORN1]][AMR_PARENT] >= 0 && (block[n][AMR_CORN1D] == n || block[n][AMR_CORN1D] == -100)){
			if (block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN1] == block[block[n][AMR_CORN1]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN1]][AMR_PARENT], BS_1, 0, 0, BS_3 + 2 * D3, send_E3_corn1, E,
					Bufferp, &(BuffersendE3corn1[nl[n]]), &(boundevent[nl[n]][301]));
				if (block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][301],0);
						rc += MPI_Isend(&BuffersendE3corn1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E3_corn1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN1]][AMR_PARENT]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN2] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == n || (block[n][AMR_CORN2D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN2]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN2], BS_1, BS_2, 0, BS_3 + D3, send_E3_corn2, E, Bufferp, &(BuffersendE3corn2[nl[n]]),
				&(boundevent[nl[n]][452]));
			if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][452],0);
					rc += MPI_Isend(&BuffersendE3corn2[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E3_corn2[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == block[n][AMR_CORN2] || (block[n][AMR_CORN2D] == -100 && block[block[n][AMR_CORN2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE3corn4[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][454]);
				}
			}
			if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_1] == block[block[n][AMR_CORN2]][AMR_CHILD1] || block[n][AMR_CORN2D_1] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn4_7[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][304]);
			}
			if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_2] == block[block[n][AMR_CORN2]][AMR_CHILD2] || block[n][AMR_CORN2D_2] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn4_8[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][354]);
			}
		}
		else{
			if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == block[n][AMR_CORN2] || (block[n][AMR_CORN2D] == -100 && block[block[n][AMR_CORN2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E3_corn4[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN2]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][454]);
				}
			}
			if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_1] == block[block[n][AMR_CORN2]][AMR_CHILD1] || block[n][AMR_CORN2D_1] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn4_1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][304]);
			}
			if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1 && (block[n][AMR_CORN2D_2] == block[block[n][AMR_CORN2]][AMR_CHILD2] || block[n][AMR_CORN2D_2] == -100) && block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn4_2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][354]);
			}
		}
		if (block[block[n][AMR_CORN2]][AMR_PARENT] >= 0 && (block[n][AMR_CORN2D] == n || block[n][AMR_CORN2D] == -100)){
			if (block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN2] == block[block[n][AMR_CORN2]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN2]][AMR_PARENT], BS_1, BS_2, 0, BS_3 + 2 * D3, send_E3_corn2, E,
					Bufferp, &(BuffersendE3corn2[nl[n]]), &(boundevent[nl[n]][302]));
				if (block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][302],0);
						rc += MPI_Isend(&BuffersendE3corn2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E3_corn2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN2]][AMR_PARENT]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == n || (block[n][AMR_CORN3D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN3]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN3], 0, BS_2, 0, BS_3 + D3, send_E3_corn3, E, Bufferp, &(BuffersendE3corn3[nl[n]]),
				&(boundevent[nl[n]][453]));
			if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][453],0);
					rc += MPI_Isend(&BuffersendE3corn3[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E3_corn3[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == block[n][AMR_CORN3] || (block[n][AMR_CORN3D] == -100 && block[block[n][AMR_CORN3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE3corn1[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][451]);
				}
			}
			if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_1] == block[block[n][AMR_CORN3]][AMR_CHILD5] || block[n][AMR_CORN3D_1] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn1_3[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][301]);
			}
			if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_2] == block[block[n][AMR_CORN3]][AMR_CHILD6] || block[n][AMR_CORN3D_2] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn1_4[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][351]);
			}
		}
		else{
			if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == block[n][AMR_CORN3] || (block[n][AMR_CORN3D] == -100 && block[block[n][AMR_CORN3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E3_corn1[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN3]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][451]);
				}
			}
			if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_1] == block[block[n][AMR_CORN3]][AMR_CHILD5] || block[n][AMR_CORN3D_1] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn1_1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][301]);
			}
			if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1 && (block[n][AMR_CORN3D_2] == block[block[n][AMR_CORN3]][AMR_CHILD6] || block[n][AMR_CORN3D_2] == -100) && block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn1_2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE], (1 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][351]);
			}
		}
		if (block[block[n][AMR_CORN3]][AMR_PARENT] >= 0 && (block[n][AMR_CORN3D] == n || block[n][AMR_CORN3D] == -100)){
			if (block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN3] == block[block[n][AMR_CORN3]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN3]][AMR_PARENT], 0, BS_2, 0, BS_3 + 2 * D3, send_E3_corn3, E,
					Bufferp, &(BuffersendE3corn3[nl[n]]), &(boundevent[nl[n]][303]));
				if (block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][303],0);
						rc += MPI_Isend(&BuffersendE3corn3[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E3_corn3[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN3]][AMR_PARENT]][AMR_NODE], (3 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	if (block[n][AMR_CORN4] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == n || (block[n][AMR_CORN4D] == -100 && block[n][AMR_TIMELEVEL]<block[block[n][AMR_CORN4]][AMR_TIMELEVEL]))){
			pack_send_E3_corn(n, block[n][AMR_CORN4], 0, 0, 0, BS_3 + D3, send_E3_corn4, E, Bufferp, &(BuffersendE3corn4[nl[n]]),
				&(boundevent[nl[n]][454]));
			if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][454],0);
					rc += MPI_Isend(&BuffersendE3corn4[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send_E3_corn4[nl[n]][0], BS_3 + 2 * D3, MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == block[n][AMR_CORN4] || (block[n][AMR_CORN4D] == -100 && block[block[n][AMR_CORN4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&BufferrecE3corn2[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][452]);
				}
			}
			if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_1] == block[block[n][AMR_CORN4]][AMR_CHILD7] || block[n][AMR_CORN4D_1] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn2_1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][302]);
			}
			if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_2] == block[block[n][AMR_CORN4]][AMR_CHILD8] || block[n][AMR_CORN4D_2] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&BufferrecE3corn2_2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][352]);
			}
		}
		else{
			if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == block[n][AMR_CORN4] || (block[n][AMR_CORN4D] == -100 && block[block[n][AMR_CORN4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]))){
				if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive_E3_corn2[nl[n]][0], (BS_3 + 2 * D3), MPI_DOUBLE, block[block[n][AMR_CORN4]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[n][AMR_CORN4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][452]);
				}
			}
			if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_1] == block[block[n][AMR_CORN4]][AMR_CHILD7] || block[n][AMR_CORN4D_1] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn2_1[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][302]);
			}
			if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1 && (block[n][AMR_CORN4D_2] == block[block[n][AMR_CORN4]][AMR_CHILD8] || block[n][AMR_CORN4D_2] == -100) && block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE] && REF_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive_E3_corn2_2[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE], (2 * NB_LOCAL + 50 * NB_LOCAL + block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][352]);
			}
		}
		if (block[block[n][AMR_CORN4]][AMR_PARENT] >= 0 && (block[n][AMR_CORN4D] == n || block[n][AMR_CORN4D] == -100)){
			if (block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n][AMR_PARENT]][AMR_CORN4] == block[block[n][AMR_CORN4]][AMR_PARENT]){
				//send to coarser grid
				pack_send_E3_corn_course(n, block[block[n][AMR_CORN4]][AMR_PARENT], 0, 0, 0, BS_3 + 2 * D3, send_E3_corn4, E,
					Bufferp, &(BuffersendE3corn4[nl[n]]), &(boundevent[nl[n]][304]));
				if (block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][304],0);
						rc += MPI_Isend(&BuffersendE3corn4[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send_E3_corn4[nl[n]][0], (BS_3 + 2 * D3) / (1 + REF_3), MPI_DOUBLE, block[block[block[n][AMR_CORN4]][AMR_PARENT]][AMR_NODE], (4 * NB_LOCAL + 50 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
}

/*Receive boundaries for compute nodes through MPI*/
void E1_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_CORN9] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN9D] == block[n][AMR_CORN9] || (block[n][AMR_CORN9D] == -100 && block[block[n][AMR_CORN9]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN9]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN9]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][461], &Statbound[nl[n]][461]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN9], 0, BS_1, 0, BS_3, receive_E1_corn11, tempreceive_E1_corn11, NULL, E,
					Bufferp, &(BufferrecE1corn11[nl[n]]), &(tempBufferrecE1corn11[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN9], block[n][AMR_CORN9], 0, BS_1, 0, BS_3, send_E1_corn11, receive_E1_corn11, NULL, E,
					Bufferp, &(BuffersendE1corn11[nl[block[n][AMR_CORN9]]]), &(BufferrecE1corn11[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN9]]][461]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN9]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN9D_1] == block[block[n][AMR_CORN9]][AMR_CHILD3] || block[n][AMR_CORN9D_1] == -100){
				if (block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][311], &Statbound[nl[n]][311]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN9]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), 0, BS_3, receive_E1_corn11_1, tempreceive_E1_corn11_1, receive_E1_corn11_12, E,
						Bufferp, &(BufferrecE1corn11_2[nl[n]]), &(tempBufferrecE1corn11_2[nl[n]]), &(BufferrecE1corn11_22[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN9]][AMR_CHILD3], block[block[n][AMR_CORN9]][AMR_CHILD3], 0, (BS_1) / (1 + REF_1), 0, BS_3, send_E1_corn11, receive_E1_corn11_1, receive_E1_corn11_12, E,
						Bufferp, &(BuffersendE1corn11[nl[block[block[n][AMR_CORN9]][AMR_CHILD3]]]), &(BufferrecE1corn11_2[nl[n]]), &(BufferrecE1corn11_22[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN9]][AMR_CHILD3]]][311]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN9D_2] == block[block[n][AMR_CORN9]][AMR_CHILD7] || block[n][AMR_CORN9D_2] == -100)){
				if (block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][361], &Statbound[nl[n]][361]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN9]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, BS_3, receive_E1_corn11_2, tempreceive_E1_corn11_2, receive_E1_corn11_22, E,
						Bufferp, &(BufferrecE1corn11_6[nl[n]]), &(tempBufferrecE1corn11_6[nl[n]]), &(BufferrecE1corn11_62[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN9]][AMR_CHILD7], block[block[n][AMR_CORN9]][AMR_CHILD7], BS_1 / (1 + REF_1), BS_1, 0, BS_3, send_E1_corn11, receive_E1_corn11_2, receive_E1_corn11_22, E,
						Bufferp, &(BuffersendE1corn11[nl[block[block[n][AMR_CORN9]][AMR_CHILD7]]]), &(BufferrecE1corn11_6[nl[n]]), &(BufferrecE1corn11_62[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN9]][AMR_CHILD7]]][311]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN10] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN10D] == block[n][AMR_CORN10] || (block[n][AMR_CORN10D] == -100 && block[block[n][AMR_CORN10]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN10]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN10]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][462], &Statbound[nl[n]][462]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN10], 0, BS_1, BS_2, BS_3, receive_E1_corn12, tempreceive_E1_corn12, NULL, E,
					Bufferp, &(BufferrecE1corn12[nl[n]]), &(tempBufferrecE1corn12[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN10], block[n][AMR_CORN10], 0, BS_1, BS_2, BS_3, send_E1_corn12, receive_E1_corn12, NULL, E,
					Bufferp, &(BuffersendE1corn12[nl[block[n][AMR_CORN10]]]), &(BufferrecE1corn12[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN10]]][462]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN10]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN10D_1] == block[block[n][AMR_CORN10]][AMR_CHILD1] || block[n][AMR_CORN10D_1] == -100){
				if (block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][312], &Statbound[nl[n]][312]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN10]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), BS_2, BS_3, receive_E1_corn12_1, tempreceive_E1_corn12_1, receive_E1_corn12_12, E,
						Bufferp, &(BufferrecE1corn12_4[nl[n]]), &(tempBufferrecE1corn12_4[nl[n]]), &(BufferrecE1corn12_42[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN10]][AMR_CHILD1], block[block[n][AMR_CORN10]][AMR_CHILD1], 0, (BS_1) / (1 + REF_1), BS_2, BS_3, send_E1_corn12, receive_E1_corn12_1, receive_E1_corn12_12, E,
						Bufferp, &(BuffersendE1corn12[nl[block[block[n][AMR_CORN10]][AMR_CHILD1]]]), &(BufferrecE1corn12_4[nl[n]]), &(BufferrecE1corn12_42[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN10]][AMR_CHILD1]]][312]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN10D_2] == block[block[n][AMR_CORN10]][AMR_CHILD5] || block[n][AMR_CORN10D_2] == -100)){
				if (block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN10]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][362], &Statbound[nl[n]][362]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN10]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_3, receive_E1_corn12_2, tempreceive_E1_corn12_2, receive_E1_corn12_22, E,
						Bufferp, &(BufferrecE1corn12_8[nl[n]]), &(tempBufferrecE1corn12_8[nl[n]]), &(BufferrecE1corn12_82[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN10]][AMR_CHILD5], block[block[n][AMR_CORN10]][AMR_CHILD5], BS_1 / (1 + REF_1), BS_1, BS_2, BS_3, send_E1_corn12, receive_E1_corn12_2, receive_E1_corn12_22, E,
						Bufferp, &(BuffersendE1corn12[nl[block[block[n][AMR_CORN10]][AMR_CHILD5]]]), &(BufferrecE1corn12_8[nl[n]]), &(BufferrecE1corn12_82[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN10]][AMR_CHILD5]]][312]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN11] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN11D] == block[n][AMR_CORN11] || (block[n][AMR_CORN11D] == -100 && block[block[n][AMR_CORN11]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN11]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN11]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][459], &Statbound[nl[n]][459]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN11], 0, BS_1, BS_2, 0, receive_E1_corn9, tempreceive_E1_corn9, NULL, E,
					Bufferp, &(BufferrecE1corn9[nl[n]]), &(tempBufferrecE1corn9[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN11], block[n][AMR_CORN11], 0, BS_1, BS_2, 0, send_E1_corn9, receive_E1_corn9, NULL, E,
					Bufferp, &(BuffersendE1corn9[nl[block[n][AMR_CORN11]]]), &(BufferrecE1corn9[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN11]]][459]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN11]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN11D_1] == block[block[n][AMR_CORN11]][AMR_CHILD2] || block[n][AMR_CORN11D_1] == -100){
				if (block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][309], &Statbound[nl[n]][309]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN11]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), BS_2, 0, receive_E1_corn9_1, tempreceive_E1_corn9_1, receive_E1_corn9_12, E,
						Bufferp, &(BufferrecE1corn9_3[nl[n]]), &(tempBufferrecE1corn9_3[nl[n]]), &(BufferrecE1corn9_32[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN11]][AMR_CHILD2], block[block[n][AMR_CORN11]][AMR_CHILD2], 0, (BS_1) / (1 + REF_1), BS_2, 0, send_E1_corn9, receive_E1_corn9_1, receive_E1_corn9_12, E,
						Bufferp, &(BuffersendE1corn9[nl[block[block[n][AMR_CORN11]][AMR_CHILD2]]]), &(BufferrecE1corn9_3[nl[n]]), &(BufferrecE1corn9_32[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN11]][AMR_CHILD2]]][309]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN11D_2] == block[block[n][AMR_CORN11]][AMR_CHILD6] || block[n][AMR_CORN11D_2] == -100)){
				if (block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN11]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][359], &Statbound[nl[n]][359]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN11]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, 0, receive_E1_corn9_2, tempreceive_E1_corn9_2, receive_E1_corn9_22, E,
						Bufferp, &(BufferrecE1corn9_7[nl[n]]), &(tempBufferrecE1corn9_7[nl[n]]), &(BufferrecE1corn9_72[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN11]][AMR_CHILD6], block[block[n][AMR_CORN11]][AMR_CHILD6], BS_1 / (1 + REF_1), BS_1, BS_2, 0, send_E1_corn9, receive_E1_corn9_2, receive_E1_corn9_22, E,
						Bufferp, &(BuffersendE1corn9[nl[block[block[n][AMR_CORN11]][AMR_CHILD6]]]), &(BufferrecE1corn9_7[nl[n]]), &(BufferrecE1corn9_72[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN11]][AMR_CHILD6]]][309]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN12] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN12D] == block[n][AMR_CORN12] || (block[n][AMR_CORN12D] == -100 && block[block[n][AMR_CORN12]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN12]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN12]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][460], &Statbound[nl[n]][460]);
				}
				unpack_receive_E1_corn(n, n, block[n][AMR_CORN12], 0, BS_1, 0, 0, receive_E1_corn10, tempreceive_E1_corn10, NULL, E,
					Bufferp, &(BufferrecE1corn10[nl[n]]), &(tempBufferrecE1corn10[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E1_corn(n, block[n][AMR_CORN12], block[n][AMR_CORN12], 0, BS_1, 0, 0, send_E1_corn10, receive_E1_corn10, NULL, E,
					Bufferp, &(BuffersendE1corn10[nl[block[n][AMR_CORN12]]]), &(BufferrecE1corn10[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN12]]][460]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN12]][AMR_REFINED] == 1 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
			//receive from finer grid
			if (block[n][AMR_CORN12D_1] == block[block[n][AMR_CORN12]][AMR_CHILD4] || block[n][AMR_CORN12D_1] == -100){
				if (block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][310], &Statbound[nl[n]][310]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN12]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), 0, 0, receive_E1_corn10_1, tempreceive_E1_corn10_1, receive_E1_corn10_12, E,
						Bufferp, &(BufferrecE1corn10_1[nl[n]]), &(tempBufferrecE1corn10_1[nl[n]]), &(BufferrecE1corn10_12[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN12]][AMR_CHILD4], block[block[n][AMR_CORN12]][AMR_CHILD4], 0, (BS_1) / (1 + REF_1), 0, 0, send_E1_corn10, receive_E1_corn10_1, receive_E1_corn10_12, E,
						Bufferp, &(BuffersendE1corn10[nl[block[block[n][AMR_CORN12]][AMR_CHILD4]]]), &(BufferrecE1corn10_1[nl[n]]), &(BufferrecE1corn10_12[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN12]][AMR_CHILD4]]][310]), calc_corr);
				}
			}
			if (REF_1 == 1 && (block[n][AMR_CORN12D_2] == block[block[n][AMR_CORN12]][AMR_CHILD8] || block[n][AMR_CORN12D_2] == -100)){
				if (block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN12]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][360], &Statbound[nl[n]][360]);
					}
					unpack_receive_E1_corn(n, n, block[block[n][AMR_CORN12]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 0, receive_E1_corn10_2, tempreceive_E1_corn10_2, receive_E1_corn10_22, E,
						Bufferp, &(BufferrecE1corn10_5[nl[n]]), &(tempBufferrecE1corn10_5[nl[n]]), &(BufferrecE1corn10_52[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E1_corn(n, block[block[n][AMR_CORN12]][AMR_CHILD8], block[block[n][AMR_CORN12]][AMR_CHILD8], BS_1 / (1 + REF_1), BS_1, 0, 0, send_E1_corn10, receive_E1_corn10_2, receive_E1_corn10_22, E,
						Bufferp, &(BuffersendE1corn10[nl[block[block[n][AMR_CORN12]][AMR_CHILD8]]]), &(BufferrecE1corn10_5[nl[n]]), &(BufferrecE1corn10_52[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN12]][AMR_CHILD8]]][310]), calc_corr);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E2_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_CORN5] >= 0){
		if (block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN5D] == block[n][AMR_CORN5] || (block[n][AMR_CORN5D] == -100 && block[block[n][AMR_CORN5]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN5]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][457], &Statbound[nl[n]][457]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN5], BS_1, 0, BS_2, 0, receive_E2_corn7, tempreceive_E2_corn7, NULL, E,
					Bufferp, &(BufferrecE2corn7[nl[n]]), &(tempBufferrecE2corn7[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN5], block[n][AMR_CORN5], BS_1, 0, BS_2, 0, send_E2_corn7, receive_E2_corn7, NULL, E,
					Bufferp, &(BuffersendE2corn7[nl[block[n][AMR_CORN5]]]), &(BufferrecE2corn7[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN5]]][457]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN5D_1] == block[block[n][AMR_CORN5]][AMR_CHILD2] || block[n][AMR_CORN5D_1] == -100){
				if (block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][307], &Statbound[nl[n]][307]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN5]][AMR_CHILD2], BS_1, 0, (BS_2) / (1 + REF_2), 0, receive_E2_corn7_1, tempreceive_E2_corn7_1, receive_E2_corn7_12, E,
						Bufferp, &(BufferrecE2corn7_5[nl[n]]), &(tempBufferrecE2corn7_5[nl[n]]), &(BufferrecE2corn7_52[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN5]][AMR_CHILD2], block[block[n][AMR_CORN5]][AMR_CHILD2], BS_1, 0, (BS_2) / (1 + REF_2), 0, send_E2_corn7, receive_E2_corn7_1, receive_E2_corn7_12, E,
						Bufferp, &(BuffersendE2corn7[nl[block[block[n][AMR_CORN5]][AMR_CHILD2]]]), &(BufferrecE2corn7_5[nl[n]]), &(BufferrecE2corn7_52[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN5]][AMR_CHILD2]]][307]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN5D_2] == block[block[n][AMR_CORN5]][AMR_CHILD4] || block[n][AMR_CORN5D_2] == -100)){
				if (block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][357], &Statbound[nl[n]][357]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN5]][AMR_CHILD4], BS_1, BS_2 / (1 + REF_2), BS_2, 0, receive_E2_corn7_2, tempreceive_E2_corn7_2, receive_E2_corn7_22, E,
						Bufferp, &(BufferrecE2corn7_7[nl[n]]), &(tempBufferrecE2corn7_7[nl[n]]), &(BufferrecE2corn7_72[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN5]][AMR_CHILD4], block[block[n][AMR_CORN5]][AMR_CHILD4], BS_1, BS_2 / (1 + REF_2), BS_2, 0, send_E2_corn7, receive_E2_corn7_2, receive_E2_corn7_22, E,
						Bufferp, &(BuffersendE2corn7[nl[block[block[n][AMR_CORN5]][AMR_CHILD4]]]), &(BufferrecE2corn7_7[nl[n]]), &(BufferrecE2corn7_72[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN5]][AMR_CHILD4]]][307]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN6] >= 0){
		if (block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN6D] == block[n][AMR_CORN6] || (block[n][AMR_CORN6D] == -100 && block[block[n][AMR_CORN6]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN6]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][458], &Statbound[nl[n]][458]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN6], BS_1, 0, BS_2, BS_3, receive_E2_corn8, tempreceive_E2_corn8, NULL, E,
					Bufferp, &(BufferrecE2corn8[nl[n]]), &(tempBufferrecE2corn8[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN6], block[n][AMR_CORN6], BS_1, 0, BS_2, BS_3, send_E2_corn8, receive_E2_corn8, NULL, E,
					Bufferp, &(BuffersendE2corn8[nl[block[n][AMR_CORN6]]]), &(BufferrecE2corn8[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN6]]][458]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN6D_1] == block[block[n][AMR_CORN6]][AMR_CHILD1] || block[n][AMR_CORN6D_1] == -100){
				if (block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][308], &Statbound[nl[n]][308]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN6]][AMR_CHILD1], BS_1, 0, (BS_2) / (1 + REF_2), BS_3, receive_E2_corn8_1, tempreceive_E2_corn8_1, receive_E2_corn8_12, E,
						Bufferp, &(BufferrecE2corn8_6[nl[n]]), &(tempBufferrecE2corn8_6[nl[n]]), &(BufferrecE2corn8_62[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN6]][AMR_CHILD1], block[block[n][AMR_CORN6]][AMR_CHILD1], BS_1, 0, (BS_2) / (1 + REF_2), BS_3, send_E2_corn8, receive_E2_corn8_1, receive_E2_corn8_12, E,
						Bufferp, &(BuffersendE2corn8[nl[block[block[n][AMR_CORN6]][AMR_CHILD1]]]), &(BufferrecE2corn8_6[nl[n]]), &(BufferrecE2corn8_62[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN6]][AMR_CHILD1]]][308]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN6D_2] == block[block[n][AMR_CORN6]][AMR_CHILD3] || block[n][AMR_CORN6D_2] == -100)){
				if (block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN6]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][358], &Statbound[nl[n]][358]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN6]][AMR_CHILD3], BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, receive_E2_corn8_2, tempreceive_E2_corn8_2, receive_E2_corn8_22, E,
						Bufferp, &(BufferrecE2corn8_8[nl[n]]), &(tempBufferrecE2corn8_8[nl[n]]), &(BufferrecE2corn8_82[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN6]][AMR_CHILD3], block[block[n][AMR_CORN6]][AMR_CHILD3], BS_1, BS_2 / (1 + REF_2), BS_2, BS_3, send_E2_corn8, receive_E2_corn8_2, receive_E2_corn8_22, E,
						Bufferp, &(BuffersendE2corn8[nl[block[block[n][AMR_CORN6]][AMR_CHILD3]]]), &(BufferrecE2corn8_8[nl[n]]), &(BufferrecE2corn8_82[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN6]][AMR_CHILD3]]][308]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN7] >= 0){
		if (block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN7D] == block[n][AMR_CORN7] || (block[n][AMR_CORN7D] == -100 && block[block[n][AMR_CORN7]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN7]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN7]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][455], &Statbound[nl[n]][455]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN7], 0, 0, BS_2, BS_3, receive_E2_corn5, tempreceive_E2_corn5, NULL, E,
					Bufferp, &(BufferrecE2corn5[nl[n]]), &(tempBufferrecE2corn5[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN7], block[n][AMR_CORN7], 0, 0, BS_2, BS_3, send_E2_corn5, receive_E2_corn5, NULL, E,
					Bufferp, &(BuffersendE2corn5[nl[block[n][AMR_CORN7]]]), &(BufferrecE2corn5[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN7]]][455]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN7]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN7D_1] == block[block[n][AMR_CORN7]][AMR_CHILD5] || block[n][AMR_CORN7D_1] == -100){
				if (block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][305], &Statbound[nl[n]][305]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN7]][AMR_CHILD5], 0, 0, (BS_2) / (1 + REF_2), BS_3, receive_E2_corn5_1, tempreceive_E2_corn5_1, receive_E2_corn5_12, E,
						Bufferp, &(BufferrecE2corn5_2[nl[n]]), &(tempBufferrecE2corn5_2[nl[n]]), &(BufferrecE2corn5_22[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN7]][AMR_CHILD5], block[block[n][AMR_CORN7]][AMR_CHILD5], 0, 0, (BS_2) / (1 + REF_2), BS_3, send_E2_corn5, receive_E2_corn5_1, receive_E2_corn5_12, E,
						Bufferp, &(BuffersendE2corn5[nl[block[block[n][AMR_CORN7]][AMR_CHILD5]]]), &(BufferrecE2corn5_2[nl[n]]), &(BufferrecE2corn5_22[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN7]][AMR_CHILD5]]][305]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN7D_2] == block[block[n][AMR_CORN7]][AMR_CHILD7] || block[n][AMR_CORN7D_2] == -100)){
				if (block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN7]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][355], &Statbound[nl[n]][355]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN7]][AMR_CHILD7], 0, BS_2 / (1 + REF_2), BS_2, BS_3, receive_E2_corn5_2, tempreceive_E2_corn5_2, receive_E2_corn5_22, E,
						Bufferp, &(BufferrecE2corn5_4[nl[n]]), &(tempBufferrecE2corn5_4[nl[n]]), &(BufferrecE2corn5_42[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN7]][AMR_CHILD7], block[block[n][AMR_CORN7]][AMR_CHILD7], 0, BS_2 / (1 + REF_2), BS_2, BS_3, send_E2_corn5, receive_E2_corn5_2, receive_E2_corn5_22, E,
						Bufferp, &(BuffersendE2corn5[nl[block[block[n][AMR_CORN7]][AMR_CHILD7]]]), &(BufferrecE2corn5_4[nl[n]]), &(BufferrecE2corn5_42[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN7]][AMR_CHILD7]]][305]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN8] >= 0){
		if (block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN8D] == block[n][AMR_CORN8] || (block[n][AMR_CORN8D] == -100 && block[block[n][AMR_CORN8]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN8]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN8]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][456], &Statbound[nl[n]][456]);
				}
				unpack_receive_E2_corn(n, n, block[n][AMR_CORN8], 0, 0, BS_2, 0, receive_E2_corn6, tempreceive_E2_corn6, NULL, E,
					Bufferp, &(BufferrecE2corn6[nl[n]]), &(tempBufferrecE2corn6[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E2_corn(n, block[n][AMR_CORN8], block[n][AMR_CORN8], 0, 0, BS_2, 0, send_E2_corn6, receive_E2_corn6, NULL, E,
					Bufferp, &(BuffersendE2corn6[nl[block[n][AMR_CORN8]]]), &(BufferrecE2corn6[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN8]]][456]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN8]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN8D_1] == block[block[n][AMR_CORN8]][AMR_CHILD6] || block[n][AMR_CORN8D_1] == -100){
				if (block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][306], &Statbound[nl[n]][306]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN8]][AMR_CHILD6], 0, 0, (BS_2) / (1 + REF_2), 0, receive_E2_corn6_1, tempreceive_E2_corn6_1, receive_E2_corn6_12, E,
						Bufferp, &(BufferrecE2corn6_1[nl[n]]), &(tempBufferrecE2corn6_1[nl[n]]), &(BufferrecE2corn6_12[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN8]][AMR_CHILD6], block[block[n][AMR_CORN8]][AMR_CHILD6], 0, 0, (BS_2) / (1 + REF_2), 0, send_E2_corn6, receive_E2_corn6_1, receive_E2_corn6_12, E,
						Bufferp, &(BuffersendE2corn6[nl[block[block[n][AMR_CORN8]][AMR_CHILD6]]]), &(BufferrecE2corn6_1[nl[n]]), &(BufferrecE2corn6_12[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN8]][AMR_CHILD6]]][306]), calc_corr);
				}
			}
			if (REF_2 == 1 && (block[n][AMR_CORN8D_2] == block[block[n][AMR_CORN8]][AMR_CHILD8] || block[n][AMR_CORN8D_2] == -100)){
				if (block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN8]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][356], &Statbound[nl[n]][356]);
					}
					unpack_receive_E2_corn(n, n, block[block[n][AMR_CORN8]][AMR_CHILD8], 0, BS_2 / (1 + REF_2), BS_2, 0, receive_E2_corn6_2, tempreceive_E2_corn6_2, receive_E2_corn6_22, E,
						Bufferp, &(BufferrecE2corn6_3[nl[n]]), &(tempBufferrecE2corn6_3[nl[n]]), &(BufferrecE2corn6_32[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E2_corn(n, block[block[n][AMR_CORN8]][AMR_CHILD8], block[block[n][AMR_CORN8]][AMR_CHILD8], 0, BS_2 / (1 + REF_2), BS_2, 0, send_E2_corn6, receive_E2_corn6_2, receive_E2_corn6_22, E,
						Bufferp, &(BuffersendE2corn6[nl[block[block[n][AMR_CORN8]][AMR_CHILD8]]]), &(BufferrecE2corn6_3[nl[n]]), &(BufferrecE2corn6_32[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN8]][AMR_CHILD8]]][306]), calc_corr);
				}
			}
		}
	}
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void E3_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double *Bufferp[NB_LOCAL], int n, int calc_corr){
#if (MPI_enable)
	int n_print = block[AMR_coord_linear(0, 0, 0, 0)][AMR_CORN2D_1];

	//positive X1
	if (block[n][AMR_CORN1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN1D] == block[n][AMR_CORN1] || (block[n][AMR_CORN1D] == -100 && block[block[n][AMR_CORN1]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN1]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][453], &Statbound[nl[n]][453]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN1], BS_1, 0, 0, BS_3, receive_E3_corn3, tempreceive_E3_corn3, NULL, E,
					Bufferp, &(BufferrecE3corn3[nl[n]]), &(tempBufferrecE3corn3[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN1], block[n][AMR_CORN1], BS_1, 0, 0, BS_3, send_E3_corn3, receive_E3_corn3, NULL, E,
					Bufferp, &(BuffersendE3corn3[nl[block[n][AMR_CORN1]]]), &(BufferrecE3corn3[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN1]]][453]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN1]][AMR_REFINED] == 1){
			if (block[n][AMR_CORN1D_1] == block[block[n][AMR_CORN1]][AMR_CHILD3] || block[n][AMR_CORN1D_1] == -100){
				if (block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][303], &Statbound[nl[n]][303]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN1]][AMR_CHILD3], BS_1, 0, 0, (BS_3) / (1 + REF_3), receive_E3_corn3_1, tempreceive_E3_corn3_1, receive_E3_corn3_12, E,
						Bufferp, &(BufferrecE3corn3_5[nl[n]]), &(tempBufferrecE3corn3_5[nl[n]]), &(BufferrecE3corn3_52[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN1]][AMR_CHILD3], block[block[n][AMR_CORN1]][AMR_CHILD3], BS_1, 0, 0, (BS_3) / (1 + REF_3), send_E3_corn3, receive_E3_corn3_1, receive_E3_corn3_12, E,
						Bufferp, &(BuffersendE3corn3[nl[block[block[n][AMR_CORN1]][AMR_CHILD3]]]), &(BufferrecE3corn3_5[nl[n]]), &(BufferrecE3corn3_52[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN1]][AMR_CHILD3]]][303]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN1D_2] == block[block[n][AMR_CORN1]][AMR_CHILD4] || block[n][AMR_CORN1D_2] == -100)){
				if (block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][353], &Statbound[nl[n]][353]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN1]][AMR_CHILD4], BS_1, 0, BS_3 / (1 + REF_3), BS_3, receive_E3_corn3_2, tempreceive_E3_corn3_2, receive_E3_corn3_22, E,
						Bufferp, &(BufferrecE3corn3_6[nl[n]]), &(tempBufferrecE3corn3_6[nl[n]]), &(BufferrecE3corn3_62[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN1]][AMR_CHILD4], block[block[n][AMR_CORN1]][AMR_CHILD4], BS_1, 0, BS_3 / (1 + REF_3), BS_3, send_E3_corn3, receive_E3_corn3_2, receive_E3_corn3_22, E,
						Bufferp, &(BuffersendE3corn3[nl[block[block[n][AMR_CORN1]][AMR_CHILD4]]]), &(BufferrecE3corn3_6[nl[n]]), &(BufferrecE3corn3_62[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN1]][AMR_CHILD4]]][303]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN2] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN2D] == block[n][AMR_CORN2] || (block[n][AMR_CORN2D] == -100 && block[block[n][AMR_CORN2]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid

			if (block[block[n][AMR_CORN2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN2]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][454], &Statbound[nl[n]][454]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN2], BS_1, BS_2, 0, BS_3, receive_E3_corn4, tempreceive_E3_corn4, NULL, E,
					Bufferp, &(BufferrecE3corn4[nl[n]]), &(tempBufferrecE3corn4[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN2], block[n][AMR_CORN2], BS_1, BS_2, 0, BS_3, send_E3_corn4, receive_E3_corn4, NULL, E,
					Bufferp, &(BuffersendE3corn4[nl[block[n][AMR_CORN2]]]), &(BufferrecE3corn4[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN2]]][454]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN2D_1] == block[block[n][AMR_CORN2]][AMR_CHILD1] || block[n][AMR_CORN2D_1] == -100){
				if (block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][304], &Statbound[nl[n]][304]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN2]][AMR_CHILD1], BS_1, BS_2, 0, (BS_3) / (1 + REF_3), receive_E3_corn4_1, tempreceive_E3_corn4_1, receive_E3_corn4_12, E,
						Bufferp, &(BufferrecE3corn4_7[nl[n]]), &(tempBufferrecE3corn4_7[nl[n]]), &(BufferrecE3corn4_72[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN2]][AMR_CHILD1], block[block[n][AMR_CORN2]][AMR_CHILD1], BS_1, BS_2, 0, (BS_3) / (1 + REF_3), send_E3_corn4, receive_E3_corn4_1, receive_E3_corn4_12, E,
						Bufferp, &(BuffersendE3corn4[nl[block[block[n][AMR_CORN2]][AMR_CHILD1]]]), &(BufferrecE3corn4_7[nl[n]]), &(BufferrecE3corn4_72[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN2]][AMR_CHILD1]]][304]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN2D_2] == block[block[n][AMR_CORN2]][AMR_CHILD2] || block[n][AMR_CORN2D_2] == -100)){
				if (block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][354], &Statbound[nl[n]][354]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN2]][AMR_CHILD2], BS_1, BS_2, BS_3 / (1 + REF_3), BS_3, receive_E3_corn4_2, tempreceive_E3_corn4_2, receive_E3_corn4_22, E,
						Bufferp, &(BufferrecE3corn4_8[nl[n]]), &(tempBufferrecE3corn4_8[nl[n]]), &(BufferrecE3corn4_82[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN2]][AMR_CHILD2], block[block[n][AMR_CORN2]][AMR_CHILD2], BS_1, BS_2, BS_3 / (1 + REF_3), BS_3, send_E3_corn4, receive_E3_corn4_2, receive_E3_corn4_22, E,
						Bufferp, &(BuffersendE3corn4[nl[block[block[n][AMR_CORN2]][AMR_CHILD2]]]), &(BufferrecE3corn4_8[nl[n]]), &(BufferrecE3corn4_82[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN2]][AMR_CHILD2]]][304]), calc_corr);
				}
			}
		}
	}

	if (block[n][AMR_CORN3] >= 0 && block[n][AMR_POLE] != 2 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN3D] == block[n][AMR_CORN3] || (block[n][AMR_CORN3D] == -100 && block[block[n][AMR_CORN3]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid

			if (block[block[n][AMR_CORN3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN3]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][451], &Statbound[nl[n]][451]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN3], 0, BS_2, 0, BS_3, receive_E3_corn1, tempreceive_E3_corn1, NULL, E,
					Bufferp, &(BufferrecE3corn1[nl[n]]), &(tempBufferrecE3corn1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN3], block[n][AMR_CORN3], 0, BS_2, 0, BS_3, send_E3_corn1, receive_E3_corn1, NULL, E,
					Bufferp, &(BuffersendE3corn1[nl[block[n][AMR_CORN3]]]), &(BufferrecE3corn1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN3]]][451]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN3]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN3D_1] == block[block[n][AMR_CORN3]][AMR_CHILD5] || block[n][AMR_CORN3D_1] == -100){
				if (block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][301], &Statbound[nl[n]][301]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN3]][AMR_CHILD5], 0, BS_2, 0, (BS_3) / (1 + REF_3), receive_E3_corn1_1, tempreceive_E3_corn1_1, receive_E3_corn1_12, E,
						Bufferp, &(BufferrecE3corn1_3[nl[n]]), &(tempBufferrecE3corn1_3[nl[n]]), &(BufferrecE3corn1_32[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN3]][AMR_CHILD5], block[block[n][AMR_CORN3]][AMR_CHILD5], 0, BS_2, 0, (BS_3) / (1 + REF_3), send_E3_corn1, receive_E3_corn1_1, receive_E3_corn1_12, E,
						Bufferp, &(BuffersendE3corn1[nl[block[block[n][AMR_CORN3]][AMR_CHILD5]]]), &(BufferrecE3corn1_3[nl[n]]), &(BufferrecE3corn1_32[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN3]][AMR_CHILD5]]][301]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN3D_2] == block[block[n][AMR_CORN3]][AMR_CHILD6] || block[n][AMR_CORN3D_2] == -100)){
				if (block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][351], &Statbound[nl[n]][351]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN3]][AMR_CHILD6], 0, BS_2, BS_3 / (1 + REF_3), BS_3, receive_E3_corn1_2, tempreceive_E3_corn1_2, receive_E3_corn1_22, E,
						Bufferp, &(BufferrecE3corn1_4[nl[n]]), &(tempBufferrecE3corn1_4[nl[n]]), &(BufferrecE3corn1_42[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN3]][AMR_CHILD6], block[block[n][AMR_CORN3]][AMR_CHILD6], 0, BS_2, BS_3 / (1 + REF_3), BS_3, send_E3_corn1, receive_E3_corn1_2, receive_E3_corn1_22, E,
						Bufferp, &(BuffersendE3corn1[nl[block[block[n][AMR_CORN3]][AMR_CHILD6]]]), &(BufferrecE3corn1_4[nl[n]]), &(BufferrecE3corn1_42[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN3]][AMR_CHILD6]]][301]), calc_corr);
				}
			}
		}
	}
	if (block[n][AMR_CORN4] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1 && (block[n][AMR_CORN4D] == block[n][AMR_CORN4] || (block[n][AMR_CORN4D] == -100 && block[block[n][AMR_CORN4]][AMR_TIMELEVEL]<block[n][AMR_TIMELEVEL]))){
			//receive from same level grid
			if (block[block[n][AMR_CORN4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_CORN4]][AMR_TIMELEVEL] - 1){
					MPI_Wait(&boundreqs[nl[n]][452], &Statbound[nl[n]][452]);
				}
				unpack_receive_E3_corn(n, n, block[n][AMR_CORN4], 0, 0, 0, BS_3, receive_E3_corn2, tempreceive_E3_corn2, NULL, E,
					Bufferp, &(BufferrecE3corn2[nl[n]]), &(tempBufferrecE3corn2[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				unpack_receive_E3_corn(n, block[n][AMR_CORN4], block[n][AMR_CORN4], 0, 0, 0, BS_3, send_E3_corn2, receive_E3_corn2, NULL, E,
					Bufferp, &(BuffersendE3corn2[nl[block[n][AMR_CORN4]]]), &(BufferrecE3corn2[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_CORN4]]][452]), calc_corr);
			}
		}
		if (block[block[n][AMR_CORN4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[n][AMR_CORN4D_1] == block[block[n][AMR_CORN4]][AMR_CHILD7] || block[n][AMR_CORN4D_1] == -100){
				if (block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][302], &Statbound[nl[n]][302]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN4]][AMR_CHILD7], 0, 0, 0, BS_3 / (1 + REF_3), receive_E3_corn2_1, tempreceive_E3_corn2_1, receive_E3_corn2_12, E,
						Bufferp, &(BufferrecE3corn2_1[nl[n]]), &(tempBufferrecE3corn2_1[nl[n]]), &(BufferrecE3corn2_12[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN4]][AMR_CHILD7], block[block[n][AMR_CORN4]][AMR_CHILD7], 0, 0, 0, BS_3 / (1 + REF_3), send_E3_corn2, receive_E3_corn2_1, receive_E3_corn2_12, E,
						Bufferp, &(BuffersendE3corn2[nl[block[block[n][AMR_CORN4]][AMR_CHILD7]]]), &(BufferrecE3corn2_1[nl[n]]), &(BufferrecE3corn2_12[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN4]][AMR_CHILD7]]][302]), calc_corr);
				}
			}
			if (REF_3 == 1 && (block[n][AMR_CORN4D_2] == block[block[n][AMR_CORN4]][AMR_CHILD8] || block[n][AMR_CORN4D_2] == -100)){
				if (block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL]) == 2 * block[block[block[n][AMR_CORN4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1){
						MPI_Wait(&boundreqs[nl[n]][352], &Statbound[nl[n]][352]);
					}
					unpack_receive_E3_corn(n, n, block[block[n][AMR_CORN4]][AMR_CHILD8], 0, 0, BS_3 / (1 + REF_3), BS_3, receive_E3_corn2_2, tempreceive_E3_corn2_2, receive_E3_corn2_22, E,
						Bufferp, &(BufferrecE3corn2_2[nl[n]]), &(tempBufferrecE3corn2_2[nl[n]]), &(BufferrecE3corn2_22[nl[n]]), NULL, calc_corr);
				}
				else{
					unpack_receive_E3_corn(n, block[block[n][AMR_CORN4]][AMR_CHILD8], block[block[n][AMR_CORN4]][AMR_CHILD8], 0, 0, BS_3 / (1 + REF_3), BS_3, send_E3_corn2, receive_E3_corn2_2, receive_E3_corn2_22, E,
						Bufferp, &(BuffersendE3corn2[nl[block[block[n][AMR_CORN4]][AMR_CHILD8]]]), &(BufferrecE3corn2_2[nl[n]]), &(BufferrecE3corn2_22[nl[n]]), &(boundevent[nl[block[block[n][AMR_CORN4]][AMR_CHILD8]]][302]), calc_corr);
				}
			}
		}
	}
#endif
}