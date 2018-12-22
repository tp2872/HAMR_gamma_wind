#include "decs_MPI.h"

/*Send boundaries of fluxes between compute nodes through MPI*/
void flux_send1(double(*restrict F1[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
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
					rc += MPI_Isend(&Buffersend2flux[nl[n]][0], NPR*BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((12 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send2_flux[nl[n]][0], NPR*BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((12 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		if (gpu == 1){
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec4flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][140]);
				}
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec4_5flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_1]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][145]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL] - 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec4_6flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_2]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][146]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&Bufferrec4_7flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_3]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][147]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL] - 1 && ref_2 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec4_8flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_4]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][148]);
			}
		}
		else{
			if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive4_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][140]);
				}
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive4_5flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_1]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][145]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL] - 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive4_6flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_2]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][146]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&receive4_7flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_3]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][147]);
			}
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL] - 1 && ref_2 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive4_8flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR2_4]][AMR_NODE], ((14 * NB_LOCAL + block[block[n][AMR_NBR2_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][148]);
			}
		}
		if (block[n][AMR_NBR2P] >= 0){
			if (block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR2P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR2P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR2P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average1(n, block[n][AMR_NBR2P], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_flux, F1, &(Bufferp[nl[n]]), &(Buffersend2flux[nl[n]]),
					&(boundevent[nl[n]][120]));
				if (block[block[n][AMR_NBR2P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR2P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][120],0);
						rc += MPI_Isend(&Buffersend2flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR2P]][AMR_NODE], ((12 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send2_flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR2P]][AMR_NODE], ((12 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
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
					rc += MPI_Isend(&Buffersend4flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((14 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send4_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((14 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}

		if (gpu == 1){
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec2flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][120]);
				}
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_1flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_5]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][121]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_2flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_6]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][122]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec2_3flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_7]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][123]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL] - 1 && ref_2 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec2_4flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_8]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][124]);
			}
		}
		else{
			if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive2_flux[nl[n]][0], NPR * BS_3 * BS_2, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][120]);
				}
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_1flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_5]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][121]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_2flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_6]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][122]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE] && ref_2 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive2_3flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_7]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][123]);
			}
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL] - 1 && ref_2 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive2_4flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_2 / (1 + ref_2)), MPI_DOUBLE, block[block[n][AMR_NBR4_8]][AMR_NODE], ((12 * NB_LOCAL + block[block[n][AMR_NBR4_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][124]);
			}
		}
		if (block[n][AMR_NBR4P] >= 0){
			if (block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR4P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR4P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR4P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average1(n, block[n][AMR_NBR4P], 0, 1, 0, BS_2, 0, BS_3, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_flux, F1, &(Bufferp[nl[n]]), &(Buffersend4flux[nl[n]]),
					&(boundevent[nl[n]][140]));
				if (block[block[n][AMR_NBR4P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR4P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][140],0);
						rc += MPI_Isend(&Buffersend4flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR4P]][AMR_NODE], ((14 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send4_flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_2) / (1 + ref_2), MPI_DOUBLE, block[block[n][AMR_NBR4P]][AMR_NODE], ((14 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void flux_send2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
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
					rc += MPI_Isend(&Buffersend3flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((13 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send3_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((13 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
	
		if (gpu == 1){
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec1flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][110]);
				}
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec1_3flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_1]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][113]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL] - 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec1_4flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_2]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][114]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&Bufferrec1_7flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_5]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][117]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec1_8flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_6]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][118]);
			}
		}
		else{
			if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive1_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][110]);
				}
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive1_3flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_1]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][113]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL] - 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive1_4flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_2]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][114]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&receive1_7flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_5]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][117]);
			}
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive1_8flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR3_6]][AMR_NODE], ((11 * NB_LOCAL + block[block[n][AMR_NBR3_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][118]);
			}
		}
		if (block[n][AMR_NBR3P] >= 0){
			if (block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR3P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR3P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR3P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average2(n, block[n][AMR_NBR3P], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3, BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_flux, F2, &(Bufferp[nl[n]]), &(Buffersend3flux[nl[n]]),
					&(boundevent[nl[n]][130]));
				if (block[block[n][AMR_NBR3P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR3P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][130],0);
						rc += MPI_Isend(&Buffersend3flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR3P]][AMR_NODE], ((13 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send3_flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR3P]][AMR_NODE], ((13 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
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
					rc += MPI_Isend(&Buffersend1flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((11 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send1_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((11 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
	
		if (gpu == 1){
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec3flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][130]);
				}
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_1flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_3]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][131]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_2flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_4]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][132]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec3_5flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_7]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][135]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&Bufferrec3_6flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_8]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][136]);
			}
		}
		else{
			if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive3_flux[nl[n]][0], NPR * BS_3 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][130]);
				}
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_1flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_3]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][131]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE] && ref_3 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_2flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_4]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][132]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE] && ref_1 == 1
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive3_5flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_7]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][135]);
			}
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_3 == 1){
				rc += MPI_Irecv(&receive3_6flux[nl[n]][0], NPR*(BS_3 / (1 + ref_3))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR1_8]][AMR_NODE], ((13 * NB_LOCAL + block[block[n][AMR_NBR1_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][136]);
			}
		}
		if (block[n][AMR_NBR1P] >= 0){
			if (block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR1P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR1P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR1P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average2(n, block[n][AMR_NBR1P], 0, BS_1, 0, 1, 0, BS_3, BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_flux, F2, &(Bufferp[nl[n]]), &(Buffersend1flux[nl[n]]),
					&(boundevent[nl[n]][110]));
				if (block[block[n][AMR_NBR1P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR1P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][110],0);
						rc += MPI_Isend(&Buffersend1flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR1P]][AMR_NODE], ((11 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send1_flux[nl[n]][0], NPR*(BS_3) / (1 + ref_3)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR1P]][AMR_NODE], ((11 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void flux_send3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] > block[n][AMR_TIMELEVEL]){
			pack_send3_flux(n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D1, BS_1, BS_2, send5_flux, F3, &(Bufferp[nl[n]]), &(Buffersend5flux[nl[n]]),
				&(boundevent[nl[n]][150]), NULL);
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
				if (gpu == 1){
					cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][150],0);
					rc += MPI_Isend(&Buffersend5flux[nl[n]][0], NPR* BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((15 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send5_flux[nl[n]][0], NPR* BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((15 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}

		if (gpu == 1){
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec6flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][160]);
				}
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec6_2flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_1]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][162]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&Bufferrec6_4flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_3]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][164]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&Bufferrec6_6flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_5]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][166]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_2 == 1){
				rc += MPI_Irecv(&Bufferrec6_8flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_7]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][168]);
			}
		}
		else{
			if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive6_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][160]);
				}
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive6_2flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_1]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_1]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][162]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&receive6_4flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_3]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_3]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][164]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&receive6_6flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_5]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_5]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][166]);
			}
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_2 == 1){
				rc += MPI_Irecv(&receive6_8flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR5_7]][AMR_NODE], ((16 * NB_LOCAL + block[block[n][AMR_NBR5_7]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][168]);
			}	
		}
		if (block[n][AMR_NBR5P] >= 0){
			if (block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR5P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR5P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR5P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average3(n, block[n][AMR_NBR5P], 0, BS_1, 0, BS_2, BS_3, BS_3 + 1, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_flux, F3, &(Bufferp[nl[n]]), &(Buffersend5flux[nl[n]]),
					&(boundevent[nl[n]][150]));
				if (block[block[n][AMR_NBR5P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR5P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][150],0);
						rc += MPI_Isend(&Buffersend5flux[nl[n]][0], NPR*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR5P]][AMR_NODE], ((15 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send5_flux[nl[n]][0], NPR*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR5P]][AMR_NODE], ((15 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
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
					rc += MPI_Isend(&Buffersend6flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((16 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				else{
					rc += MPI_Isend(&send6_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((16 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
				}
				MPI_Request_free(&req[nl[n]]);
			}
		}
		
		if (gpu == 1){
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&Bufferrec5flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][150]);
				}
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&Bufferrec5_1flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_2]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][151]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&Bufferrec5_3flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_4]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][153]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&Bufferrec5_5flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_6]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][155]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_2 == 1){
				rc += MPI_Irecv(&Bufferrec5_7flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_8]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][157]);
			}
		}
		else{
			if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
				if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					rc += MPI_Irecv(&receive5_flux[nl[n]][0], NPR * BS_2 * BS_1, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][150]);
				}
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL] - 1){
				rc += MPI_Irecv(&receive5_1flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_2]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_2]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][151]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL] - 1 && ref_2 == 1){
				rc += MPI_Irecv(&receive5_3flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_4]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_4]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][153]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL] - 1 && ref_1 == 1){
				rc += MPI_Irecv(&receive5_5flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_6]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_6]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][155]);
			}
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1)set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
			if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE]
				&& block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL] - 1 && ref_1 == 1 && ref_2 == 1){
				rc += MPI_Irecv(&receive5_7flux[nl[n]][0], NPR*(BS_2 / (1 + ref_2))*(BS_1 / (1 + ref_1)), MPI_DOUBLE, block[block[n][AMR_NBR6_8]][AMR_NODE], ((15 * NB_LOCAL + block[block[n][AMR_NBR6_8]][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &boundreqs[nl[n]][157]);
			}
		}
		if (block[n][AMR_NBR6P] >= 0){
			if (block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1){
				ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_NBR6P]][AMR_LEVEL1];
				ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_NBR6P]][AMR_LEVEL2];
				ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_NBR6P]][AMR_LEVEL3];
				//send to coarser grid
				pack_send_flux_average3(n, block[n][AMR_NBR6P], 0, BS_1, 0, BS_2, 0, 1, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_flux, F3, &(Bufferp[nl[n]]), &(Buffersend6flux[nl[n]]),
					&(boundevent[nl[n]][160]));
				if (block[block[n][AMR_NBR6P]][AMR_NODE] != block[n][AMR_NODE] && block[n][AMR_NSTEP] % (2 * block[block[n][AMR_NBR6P]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6P]][AMR_TIMELEVEL] - 1){
					if (gpu == 1){
						cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][160],0);
						rc += MPI_Isend(&Buffersend6flux[nl[n]][0], NPR*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR6P]][AMR_NODE], ((16 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
					}
					else{
						rc += MPI_Isend(&send6_flux[nl[n]][0], NPR*(BS_2) / (1 + ref_2)*(BS_1) / (1 + ref_1), MPI_DOUBLE, block[block[n][AMR_NBR6P]][AMR_NODE], ((16 * NB_LOCAL + block[n][AMR_NUMBER]) % MPI_TAG_MAX), mpi_cartcomm, &req[nl[n]]);
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
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int flag;
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR4]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE2] == 0) MPI_Test(&boundreqs[nl[n]][120], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][120], &Statbound[nl[n]][120]);
					else if (block[n][AMR_IPROBE2] != 1) block[n][AMR_IPROBE2] = -1;
				}
				if (block[n][AMR_IPROBE2] == 0)unpack_receive1_flux(n, n, block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2, BS_3, receive2_flux, receive2_flux1, NULL, F1, &(Bufferp[nl[n]]), &(Bufferrec2flux[nl[n]]), &(Bufferrec2flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE2] == 0)unpack_receive1_flux(n, block[n][AMR_NBR4], block[n][AMR_NBR4], 0, 1, 0, BS_2, 0, BS_3, BS_2, BS_3, send2_flux, receive2_flux1, NULL, F1,
					&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4]]]), &(Bufferrec2flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4]]][120]), calc_corr);
			}
		}
		if (block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR4_5]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_5]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE2_1] == 0) MPI_Test(&boundreqs[nl[n]][121], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][121], &Statbound[nl[n]][121]);
					else if (block[n][AMR_IPROBE2_1] != 1) block[n][AMR_IPROBE2_1] = -1;
				}
				if (block[n][AMR_IPROBE2_1] == 0)unpack_receive1_flux(n, n, block[n][AMR_NBR4_5], 0, 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_1flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[nl[n]]), &(Bufferrec2_1flux[nl[n]]), &(Bufferrec2_1flux1[nl[n]]), &(Bufferrec2_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE2_1] == 0)unpack_receive1_flux(n, block[n][AMR_NBR4_5], block[n][AMR_NBR4_5], 0, 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_flux, receive2_1flux1, receive2_1flux2, F1,
					&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4_5]]]), &(Bufferrec2_1flux1[nl[n]]), &(Bufferrec2_1flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4_5]]][120]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR4_6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_2] == 0) MPI_Test(&boundreqs[nl[n]][122], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][122], &Statbound[nl[n]][122]);
						else if (block[n][AMR_IPROBE2_2] != 1) block[n][AMR_IPROBE2_2] = -1;
					}
					if (block[n][AMR_IPROBE2_2] == 0)unpack_receive1_flux(n, n, block[n][AMR_NBR4_6], 0, 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_2flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_2flux[nl[n]]), &(Bufferrec2_2flux1[nl[n]]), &(Bufferrec2_2flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE2_2] == 0)unpack_receive1_flux(n, block[n][AMR_NBR4_6], block[n][AMR_NBR4_6], 0, 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_flux, receive2_2flux1, receive2_2flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4_6]]]), &(Bufferrec2_2flux1[nl[n]]), &(Bufferrec2_2flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4_6]]][120]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR4_7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_3] == 0) MPI_Test(&boundreqs[nl[n]][123], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][123], &Statbound[nl[n]][123]);
						else if (block[n][AMR_IPROBE2_3] != 1) block[n][AMR_IPROBE2_3] = -1;
					}
					if (block[n][AMR_IPROBE2_3] == 0)unpack_receive1_flux(n, n, block[n][AMR_NBR4_7], 0, 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_3flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_3flux[nl[n]]), &(Bufferrec2_3flux1[nl[n]]), &(Bufferrec2_3flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE2_3] == 0)unpack_receive1_flux(n, block[n][AMR_NBR4_7], block[n][AMR_NBR4_7], 0, 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_flux, receive2_3flux1, receive2_3flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4_7]]]), &(Bufferrec2_3flux1[nl[n]]), &(Bufferrec2_3flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4_7]]][120]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1 && ref_3 == 1){
				if (block[block[n][AMR_NBR4_8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR4_8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE2_4] == 0) MPI_Test(&boundreqs[nl[n]][124], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][124], &Statbound[nl[n]][124]);
						else if (block[n][AMR_IPROBE2_4] != 1) block[n][AMR_IPROBE2_4] = -1;
					}
					if (block[n][AMR_IPROBE2_4] == 0)unpack_receive1_flux(n, n, block[n][AMR_NBR4_8], 0, 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive2_4flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec2_4flux[nl[n]]), &(Bufferrec2_4flux1[nl[n]]), &(Bufferrec2_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE2_4] == 0)unpack_receive1_flux(n, block[n][AMR_NBR4_8], block[n][AMR_NBR4_8], 0, 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send2_flux, receive2_4flux1, receive2_4flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend2flux[nl[block[n][AMR_NBR4_8]]]), &(Bufferrec2_4flux1[nl[n]]), &(Bufferrec2_4flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR4_8]]][120]), calc_corr);
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
					flag = 0;
					if (block[n][AMR_IPROBE4] == 0) MPI_Test(&boundreqs[nl[n]][140], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][140], &Statbound[nl[n]][140]);
					else if (block[n][AMR_IPROBE4] != 1) block[n][AMR_IPROBE4] = -1;
				}
				if (block[n][AMR_IPROBE4] == 0) unpack_receive1_flux(n, n, block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, receive4_flux, receive4_flux1, NULL, F1, &(Bufferp[nl[n]]), &(Bufferrec4flux[nl[n]]), &(Bufferrec4flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE4] == 0) unpack_receive1_flux(n, block[n][AMR_NBR2], block[n][AMR_NBR2], BS_1, BS_1 + 1, 0, BS_2, 0, BS_3,
					BS_2, BS_3, send4_flux, receive4_flux1, NULL, F1,
					&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2]]]), &(Bufferrec4flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2]]][140]), calc_corr);
			}
		}
		if (block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR2_1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_1]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE4_1] == 0) MPI_Test(&boundreqs[nl[n]][145], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][145], &Statbound[nl[n]][145]);
					else if (block[n][AMR_IPROBE4_1] != 1) block[n][AMR_IPROBE4_1] = -1;
				}
				if (block[n][AMR_IPROBE4_1] == 0) unpack_receive1_flux(n, n, block[n][AMR_NBR2_1], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_5flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[nl[n]]), &(Bufferrec4_5flux[nl[n]]), &(Bufferrec4_5flux1[nl[n]]), &(Bufferrec4_5flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE4_1] == 0) unpack_receive1_flux(n, block[n][AMR_NBR2_1], block[n][AMR_NBR2_1], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3),
					BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_flux, receive4_5flux1, receive4_5flux2, F1,
					&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2_1]]]), &(Bufferrec4_5flux1[nl[n]]), &(Bufferrec4_5flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2_1]]][140]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR2_2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_2]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_2] == 0) MPI_Test(&boundreqs[nl[n]][146], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][146], &Statbound[nl[n]][146]);
						else if (block[n][AMR_IPROBE4_2] != 1) block[n][AMR_IPROBE4_2] = -1;
					}
					if (block[n][AMR_IPROBE4_2] == 0) unpack_receive1_flux(n, n, block[n][AMR_NBR2_2], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_6flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_6flux[nl[n]]), &(Bufferrec4_6flux1[nl[n]]), &(Bufferrec4_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE4_2] == 0) unpack_receive1_flux(n, block[n][AMR_NBR2_2], block[n][AMR_NBR2_2], BS_1, BS_1 + 1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_flux, receive4_6flux1, receive4_6flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2_2]]]), &(Bufferrec4_6flux1[nl[n]]), &(Bufferrec4_6flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2_2]]][140]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR2_3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_3]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_3] == 0) MPI_Test(&boundreqs[nl[n]][147], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][147], &Statbound[nl[n]][147]);
						else if (block[n][AMR_IPROBE4_3] != 1) block[n][AMR_IPROBE4_3] = -1;
					}
					if (block[n][AMR_IPROBE4_3] == 0) unpack_receive1_flux(n, n, block[n][AMR_NBR2_3], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_7flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_7flux[nl[n]]), &(Bufferrec4_7flux1[nl[n]]), &(Bufferrec4_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE4_3] == 0) unpack_receive1_flux(n, block[n][AMR_NBR2_3], block[n][AMR_NBR2_3], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3),
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_flux, receive4_7flux1, receive4_7flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2_3]]]), &(Bufferrec4_7flux1[nl[n]]), &(Bufferrec4_7flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2_3]]][140]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1 && ref_3 == 1){
				if (block[block[n][AMR_NBR2_4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR2_4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE4_4] == 0) MPI_Test(&boundreqs[nl[n]][148], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][148], &Statbound[nl[n]][148]);
						else if (block[n][AMR_IPROBE4_4] != 1) block[n][AMR_IPROBE4_4] = -1;
					}
					if (block[n][AMR_IPROBE4_4] == 0) unpack_receive1_flux(n, n, block[n][AMR_NBR2_4], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), receive4_8flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[nl[n]]), &(Bufferrec4_8flux[nl[n]]), &(Bufferrec4_8flux1[nl[n]]), &(Bufferrec4_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE4_4] == 0) unpack_receive1_flux(n, block[n][AMR_NBR2_4], block[n][AMR_NBR2_4], BS_1, BS_1 + 1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3,
						BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), send4_flux, receive4_8flux1, receive4_8flux2, F1,
						&(Bufferp[nl[n]]), &(Buffersend4flux[nl[block[n][AMR_NBR2_4]]]), &(Bufferrec4_8flux1[nl[n]]), &(Bufferrec4_8flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR2_4]]][140]), calc_corr);
				}
			}
		}
	}
#endif
}

void flux_rec2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int flag;
	//Positive X2
	if (block[n][AMR_NBR1] >= 0 && block[n][AMR_POLE] != 1 && block[n][AMR_POLE] != 3){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR1]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE3] == 0) MPI_Test(&boundreqs[nl[n]][130], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][130], &Statbound[nl[n]][130]);
					else if (block[n][AMR_IPROBE3] != 1) block[n][AMR_IPROBE3] = -1;
				}
				if (block[n][AMR_IPROBE3] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, receive3_flux, receive3_flux1, NULL, F2, &(Bufferp[nl[n]]), &(Bufferrec3flux[nl[n]]), &(Bufferrec3flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE3] == 0) unpack_receive2_flux(n, block[n][AMR_NBR1], block[n][AMR_NBR1], 0, BS_1, 0, 1, 0, BS_3,
					BS_1, BS_3, send3_flux, receive3_flux1, NULL, F2,
					&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1]]]), &(Bufferrec3flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1]]][130]), calc_corr);
			}
		}
		if (block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR1_3]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_3]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE3_1] == 0) MPI_Test(&boundreqs[nl[n]][131], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][131], &Statbound[nl[n]][131]);
					else if (block[n][AMR_IPROBE3_1] != 1) block[n][AMR_IPROBE3_1] = -1;
				}
				if (block[n][AMR_IPROBE3_1] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR1_3], 0, BS_1 / (1 + ref_1), 0, 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_1flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[nl[n]]), &(Bufferrec3_1flux[nl[n]]), &(Bufferrec3_1flux1[nl[n]]), &(Bufferrec3_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE3_1] == 0) unpack_receive2_flux(n, block[n][AMR_NBR1_3], block[n][AMR_NBR1_3], 0, BS_1 / (1 + ref_1), 0, 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_flux, receive3_1flux1, receive3_1flux2, F2,
					&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1_3]]]), &(Bufferrec3_1flux1[nl[n]]), &(Bufferrec3_1flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1_3]]][130]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR1_4], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR1_4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_2] == 0) MPI_Test(&boundreqs[nl[n]][132], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][132], &Statbound[nl[n]][132]);
						else if (block[n][AMR_IPROBE3_2] != 1) block[n][AMR_IPROBE3_2] = -1;
					}
					if (block[n][AMR_IPROBE3_2] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR1_4], 0, BS_1 / (1 + ref_1), 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_2flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_2flux[nl[n]]), &(Bufferrec3_2flux1[nl[n]]), &(Bufferrec3_2flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE3_2] == 0) unpack_receive2_flux(n, block[n][AMR_NBR1_4], block[n][AMR_NBR1_4], 0, BS_1 / (1 + ref_1), 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_flux, receive3_2flux1, receive3_2flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1_4]]]), &(Bufferrec3_2flux1[nl[n]]), &(Bufferrec3_2flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1_4]]][130]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR1_7], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR1_7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_3] == 0) MPI_Test(&boundreqs[nl[n]][135], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][135], &Statbound[nl[n]][135]);
						else if (block[n][AMR_IPROBE3_3] != 1) block[n][AMR_IPROBE3_3] = -1;
					}
					if (block[n][AMR_IPROBE3_3] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR1_7], BS_1 / (1 + ref_1), BS_1, 0, 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_5flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_5flux[nl[n]]), &(Bufferrec3_5flux1[nl[n]]), &(Bufferrec3_5flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE3_3] == 0) unpack_receive2_flux(n, block[n][AMR_NBR1_7], block[n][AMR_NBR1_7], BS_1 / (1 + ref_1), BS_1, 0, 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_flux, receive3_5flux1, receive3_5flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1_7]]]), &(Bufferrec3_5flux1[nl[n]]), &(Bufferrec3_5flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1_7]]][130]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR1_8], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1 && ref_3 == 1){
				if (block[block[n][AMR_NBR1_8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR1_8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE3_4] == 0) MPI_Test(&boundreqs[nl[n]][136], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][136], &Statbound[nl[n]][136]);
						else if (block[n][AMR_IPROBE3_4] != 1) block[n][AMR_IPROBE3_4] = -1;
					}
					if (block[n][AMR_IPROBE3_4] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR1_8], BS_1 / (1 + ref_1), BS_1, 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive3_6flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec3_6flux[nl[n]]), &(Bufferrec3_6flux1[nl[n]]), &(Bufferrec3_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE3_4] == 0) unpack_receive2_flux(n, block[n][AMR_NBR1_8], block[n][AMR_NBR1_8], BS_1 / (1 + ref_1), BS_1, 0, 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send3_flux, receive3_6flux1, receive3_6flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend3flux[nl[block[n][AMR_NBR1_8]]]), &(Bufferrec3_6flux1[nl[n]]), &(Bufferrec3_6flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR1_8]]][130]), calc_corr);
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
					flag = 0;
					if (block[n][AMR_IPROBE1] == 0) MPI_Test(&boundreqs[nl[n]][110], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][110], &Statbound[nl[n]][110]);
					else if (block[n][AMR_IPROBE1] != 1) block[n][AMR_IPROBE1] = -1;
				}
				if (block[n][AMR_IPROBE1] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1, BS_3, receive1_flux, receive1_flux1, NULL, F2, &(Bufferp[nl[n]]), &(Bufferrec1flux[nl[n]]), &(Bufferrec1flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE1] == 0) unpack_receive2_flux(n, block[n][AMR_NBR3], block[n][AMR_NBR3], 0, BS_1, BS_2, BS_2 + 1, 0, BS_3,
					BS_1, BS_3, send1_flux, receive1_flux1, NULL, F2,
					&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3]]]), &(Bufferrec1flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3]]][110]), calc_corr);
			}
		}
		if (block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR3_1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_1]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE1_1] == 0) MPI_Test(&boundreqs[nl[n]][113], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][113], &Statbound[nl[n]][113]);
					else if (block[n][AMR_IPROBE1_1] != 1) block[n][AMR_IPROBE1_1] = -1;
				}
				if (block[n][AMR_IPROBE1_1] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR3_1], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_3flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[nl[n]]), &(Bufferrec1_3flux[nl[n]]), &(Bufferrec1_3flux1[nl[n]]), &(Bufferrec1_3flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE1_1] == 0) unpack_receive2_flux(n, block[n][AMR_NBR3_1], block[n][AMR_NBR3_1], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
					BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_flux, receive1_3flux1, receive1_3flux2, F2,
					&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3_1]]]), &(Bufferrec1_3flux1[nl[n]]), &(Bufferrec1_3flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3_1]]][110]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR3_2], &ref_1, &ref_2, &ref_3);
			if (ref_3 == 1){
				if (block[block[n][AMR_NBR3_2]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_2]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_2] == 0) MPI_Test(&boundreqs[nl[n]][114], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][114], &Statbound[nl[n]][114]);
						else if (block[n][AMR_IPROBE1_2] != 1) block[n][AMR_IPROBE1_2] = -1;
					}
					if (block[n][AMR_IPROBE1_2] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR3_2], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_4flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_4flux[nl[n]]), &(Bufferrec1_4flux1[nl[n]]), &(Bufferrec1_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE1_2] == 0) unpack_receive2_flux(n, block[n][AMR_NBR3_2], block[n][AMR_NBR3_2], 0, BS_1 / (1 + ref_1), BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_flux, receive1_4flux1, receive1_4flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3_2]]]), &(Bufferrec1_4flux1[nl[n]]), &(Bufferrec1_4flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3_2]]][110]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR3_5], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR3_5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_5]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_3] == 0) MPI_Test(&boundreqs[nl[n]][117], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][117], &Statbound[nl[n]][117]);
						else if (block[n][AMR_IPROBE1_3] != 1) block[n][AMR_IPROBE1_3] = -1;
					}
					if (block[n][AMR_IPROBE1_3] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR3_5], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_7flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_7flux[nl[n]]), &(Bufferrec1_7flux1[nl[n]]), &(Bufferrec1_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE1_3] == 0) unpack_receive2_flux(n, block[n][AMR_NBR3_5], block[n][AMR_NBR3_5], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, 0, BS_3 / (1 + ref_3),
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_flux, receive1_7flux1, receive1_7flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3_5]]]), &(Bufferrec1_7flux1[nl[n]]), &(Bufferrec1_7flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3_5]]][110]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR3_6], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1 && ref_3 == 1){
				if (block[block[n][AMR_NBR3_6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR3_6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE1_4] == 0) MPI_Test(&boundreqs[nl[n]][118], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][118], &Statbound[nl[n]][118]);
						else if (block[n][AMR_IPROBE1_4] != 1) block[n][AMR_IPROBE1_4] = -1;
					}
					if (block[n][AMR_IPROBE1_4] == 0) unpack_receive2_flux(n, n, block[n][AMR_NBR3_6], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), receive1_8flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[nl[n]]), &(Bufferrec1_8flux[nl[n]]), &(Bufferrec1_8flux1[nl[n]]), &(Bufferrec1_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE1_4] == 0) unpack_receive2_flux(n, block[n][AMR_NBR3_6], block[n][AMR_NBR3_6], BS_1 / (1 + ref_1), BS_1, BS_2, BS_2 + 1, BS_3 / (1 + ref_3), BS_3,
						BS_1 / (1 + ref_1), BS_3 / (1 + ref_3), send1_flux, receive1_8flux1, receive1_8flux2, F2,
						&(Bufferp[nl[n]]), &(Buffersend1flux[nl[block[n][AMR_NBR3_6]]]), &(Bufferrec1_8flux1[nl[n]]), &(Bufferrec1_8flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR3_6]]][110]), calc_corr);
				}
			}
		}
	}
#endif
}
void flux_rec3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr){
	int ref_1, ref_2, ref_3;
#if (MPI_enable)
	int flag;
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR6]][AMR_TIMELEVEL] < block[n][AMR_TIMELEVEL]){
			//receive from same level grid
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE5] == 0) MPI_Test(&boundreqs[nl[n]][150], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][150], &Statbound[nl[n]][150]);
					else if (block[n][AMR_IPROBE5] != 1) block[n][AMR_IPROBE5] = -1;
				}
				if (block[n][AMR_IPROBE5] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1, BS_2, receive5_flux, receive5_flux1, NULL, F3, &(Bufferp[nl[n]]), &(Bufferrec5flux[nl[n]]), &(Bufferrec5flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE5] == 0) unpack_receive3_flux(n, block[n][AMR_NBR6], block[n][AMR_NBR6], 0, BS_1, 0, BS_2, 0, D3,
					BS_1, BS_2, send5_flux, receive5_flux1, NULL, F3,
					&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6]]]), &(Bufferrec5flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6]]][150]), calc_corr);
			}
		}
		if (block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR6_2]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_2]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE5_1] == 0) MPI_Test(&boundreqs[nl[n]][151], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][151], &Statbound[nl[n]][151]);
					else if (block[n][AMR_IPROBE5_1] != 1) block[n][AMR_IPROBE5_1] = -1;
				}
				if (block[n][AMR_IPROBE5_1] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR6_2], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_1flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[nl[n]]), &(Bufferrec5_1flux[nl[n]]), &(Bufferrec5_1flux1[nl[n]]), &(Bufferrec5_1flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE5_1] == 0) unpack_receive3_flux(n, block[n][AMR_NBR6_2], block[n][AMR_NBR6_2], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_flux, receive5_1flux1, receive5_1flux2, F3,
					&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6_2]]]), &(Bufferrec5_1flux1[nl[n]]), &(Bufferrec5_1flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6_2]]][150]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR6_4], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR6_4]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_4]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_2] == 0) MPI_Test(&boundreqs[nl[n]][153], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][153], &Statbound[nl[n]][153]);
						else if (block[n][AMR_IPROBE5_2] != 1) block[n][AMR_IPROBE5_2] = -1;
					}
					if (block[n][AMR_IPROBE5_2] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR6_4], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_3flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_3flux[nl[n]]), &(Bufferrec5_3flux1[nl[n]]), &(Bufferrec5_3flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE5_2] == 0) unpack_receive3_flux(n, block[n][AMR_NBR6_4], block[n][AMR_NBR6_4], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_flux, receive5_3flux1, receive5_3flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6_4]]]), &(Bufferrec5_3flux1[nl[n]]), &(Bufferrec5_3flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6_4]]][150]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR6_6], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR6_6]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_6]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_3] == 0) MPI_Test(&boundreqs[nl[n]][155], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][155], &Statbound[nl[n]][155]);
						else if (block[n][AMR_IPROBE5_3] != 1) block[n][AMR_IPROBE5_3] = -1;
					}
					if (block[n][AMR_IPROBE5_3] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR6_6], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_5flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_5flux[nl[n]]), &(Bufferrec5_5flux1[nl[n]]), &(Bufferrec5_5flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE5_3] == 0) unpack_receive3_flux(n, block[n][AMR_NBR6_6], block[n][AMR_NBR6_6], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_flux, receive5_5flux1, receive5_5flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6_6]]]), &(Bufferrec5_5flux1[nl[n]]), &(Bufferrec5_5flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6_6]]][150]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR6_8], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1 && ref_2 == 1){
				if (block[block[n][AMR_NBR6_8]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR6_8]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE5_4] == 0) MPI_Test(&boundreqs[nl[n]][157], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][157], &Statbound[nl[n]][157]);
						else if (block[n][AMR_IPROBE5_4] != 1) block[n][AMR_IPROBE5_4] = -1;
					}
					if (block[n][AMR_IPROBE5_4] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR6_8], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive5_7flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec5_7flux[nl[n]]), &(Bufferrec5_7flux1[nl[n]]), &(Bufferrec5_7flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE5_4] == 0) unpack_receive3_flux(n, block[n][AMR_NBR6_8], block[n][AMR_NBR6_8], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send5_flux, receive5_7flux1, receive5_7flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend5flux[nl[block[n][AMR_NBR6_8]]]), &(Bufferrec5_7flux1[nl[n]]), &(Bufferrec5_7flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR6_8]]][150]), calc_corr);
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
					flag = 0;
					if (block[n][AMR_IPROBE6] == 0) MPI_Test(&boundreqs[nl[n]][160], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][160], &Statbound[nl[n]][160]);
					else if (block[n][AMR_IPROBE6] != 1) block[n][AMR_IPROBE6] = -1;
				}
				if (block[n][AMR_IPROBE6] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, receive6_flux, receive6_flux1, NULL, F3, &(Bufferp[nl[n]]), &(Bufferrec6flux[nl[n]]), &(Bufferrec6flux1[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE6] == 0) unpack_receive3_flux(n, block[n][AMR_NBR5], block[n][AMR_NBR5], 0, BS_1, 0, BS_2, BS_3, BS_3 + D3,
					BS_1, BS_2, send6_flux, receive6_flux1, NULL, F3,
					&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5]]]), &(Bufferrec6flux1[nl[n]]), &(NULL_POINTER[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5]]][160]), calc_corr);
			}
		}
		if (block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){
			set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
			//receive from finer grid
			if (block[block[n][AMR_NBR5_1]][AMR_NODE] != block[n][AMR_NODE]){
				if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_1]][AMR_TIMELEVEL] - 1){
					flag = 0;
					if (block[n][AMR_IPROBE6_1] == 0) MPI_Test(&boundreqs[nl[n]][162], &flag, &Statbound[nl[n]][0]);
					if (flag == 1) MPI_Wait(&boundreqs[nl[n]][162], &Statbound[nl[n]][162]);
					else if (block[n][AMR_IPROBE6_1] != 1) block[n][AMR_IPROBE6_1] = -1;
				}
				if (block[n][AMR_IPROBE6_1] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR5_1], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_2flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[nl[n]]), &(Bufferrec6_2flux[nl[n]]), &(Bufferrec6_2flux1[nl[n]]), &(Bufferrec6_2flux2[nl[n]]), NULL, calc_corr);
			}
			else{
				if (block[n][AMR_IPROBE6_1] == 0) unpack_receive3_flux(n, block[n][AMR_NBR5_1], block[n][AMR_NBR5_1], 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
					BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_flux, receive6_2flux1, receive6_2flux2, F3,
					&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5_1]]]), &(Bufferrec6_2flux1[nl[n]]), &(Bufferrec6_2flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5_1]]][160]), calc_corr);
			}
			set_ref(n, block[n][AMR_NBR5_3], &ref_1, &ref_2, &ref_3);
			if (ref_2 == 1){
				if (block[block[n][AMR_NBR5_3]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_3]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_2] == 0) MPI_Test(&boundreqs[nl[n]][164], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][164], &Statbound[nl[n]][164]);
						else if (block[n][AMR_IPROBE6_2] != 1) block[n][AMR_IPROBE6_2] = -1;
					}
					if (block[n][AMR_IPROBE6_2] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR5_3], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_4flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_4flux[nl[n]]), &(Bufferrec6_4flux1[nl[n]]), &(Bufferrec6_4flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE6_2] == 0) unpack_receive3_flux(n, block[n][AMR_NBR5_3], block[n][AMR_NBR5_3], 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_flux, receive6_4flux1, receive6_4flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5_3]]]), &(Bufferrec6_4flux1[nl[n]]), &(Bufferrec6_4flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5_3]]][160]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR5_5], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1){
				if (block[block[n][AMR_NBR5_5]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_5]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_3] == 0) MPI_Test(&boundreqs[nl[n]][166], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][166], &Statbound[nl[n]][166]);
						else if (block[n][AMR_IPROBE6_3] != 1) block[n][AMR_IPROBE6_3] = -1;
					}
					if (block[n][AMR_IPROBE6_3] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR5_5], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_6flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_6flux[nl[n]]), &(Bufferrec6_6flux1[nl[n]]), &(Bufferrec6_6flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE6_3] == 0) unpack_receive3_flux(n, block[n][AMR_NBR5_5], block[n][AMR_NBR5_5], BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_flux, receive6_6flux1, receive6_6flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5_5]]]), &(Bufferrec6_6flux1[nl[n]]), &(Bufferrec6_6flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5_5]]][160]), calc_corr);
				}
			}
			set_ref(n, block[n][AMR_NBR5_7], &ref_1, &ref_2, &ref_3);
			if (ref_1 == 1 && ref_2 == 1){
				if (block[block[n][AMR_NBR5_7]][AMR_NODE] != block[n][AMR_NODE]){
					if ((calc_corr == 1 || calc_corr == 5) && nstep % (2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL]) == 2 * block[block[n][AMR_NBR5_7]][AMR_TIMELEVEL] - 1){
						flag = 0;
						if (block[n][AMR_IPROBE6_4] == 0) MPI_Test(&boundreqs[nl[n]][168], &flag, &Statbound[nl[n]][0]);
						if (flag == 1) MPI_Wait(&boundreqs[nl[n]][168], &Statbound[nl[n]][168]);
						else if (block[n][AMR_IPROBE6_4] != 1) block[n][AMR_IPROBE6_4] = -1;
					}
					if (block[n][AMR_IPROBE6_4] == 0) unpack_receive3_flux(n, n, block[n][AMR_NBR5_7], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), receive6_8flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[nl[n]]), &(Bufferrec6_8flux[nl[n]]), &(Bufferrec6_8flux1[nl[n]]), &(Bufferrec6_8flux2[nl[n]]), NULL, calc_corr);
				}
				else{
					if (block[n][AMR_IPROBE6_4] == 0) unpack_receive3_flux(n, block[n][AMR_NBR5_7], block[n][AMR_NBR5_7], BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, BS_3, BS_3 + D3,
						BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), send6_flux, receive6_8flux1, receive6_8flux2, F3,
						&(Bufferp[nl[n]]), &(Buffersend6flux[nl[block[n][AMR_NBR5_7]]]), &(Bufferrec6_8flux1[nl[n]]), &(Bufferrec6_8flux2[nl[n]]), &(boundevent[nl[block[n][AMR_NBR5_7]]][160]), calc_corr);
				}
			}
		}
	}
#endif
}