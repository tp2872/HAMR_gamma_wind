#include "decs_MPI.h"

/*Send boundaries between compute nodes through MPI*/
void bound_send1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep){
#if (MPI_enable)
	//Exchange boundary cells for MPI threads
	//Positive X1
	int cond1, cond2;
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR2]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL] && block[n][AMR_NSTEP]%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[n][AMR_NBR2], BS_1 - N1G, BS_1, -N2G, BS_2 + N2G, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), send2, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2[nl[n]]),
				&(boundevent1[nl[n]][20]), &(boundevent2[nl[n]][20]));
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][40]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][20],0);
					if (cond1) rc += MPI_Isend(&Buffersend2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][40]);
					if (cond1) rc += MPI_Isend(&send2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR2]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send2_1 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], BS_1 - N1G, BS_1, -N2G, BS_2 / (1 + REF_2) + N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_1, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_1[nl[n]]),
				&(boundevent1[nl[n]][21]), &(boundevent2[nl[n]][21]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec4_5[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][45]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][21],0);
					if (cond1) rc += MPI_Isend(&Buffersend2_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (21 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive4_5[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][45]);
					if (cond1) rc += MPI_Isend(&send2_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE], (21 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send2_2 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], BS_1 - N1G, BS_1, -N2G, BS_2 / (1 + REF_2) + N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_2, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_2[nl[n]]),
				&(boundevent1[nl[n]][22]), &(boundevent2[nl[n]][22]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec4_6[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][46]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][22],0);
					if (cond1) rc += MPI_Isend(&Buffersend2_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (22 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive4_6[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][46]);
					if (cond1) rc += MPI_Isend(&send2_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE], (22 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send2_3 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], BS_1 - N1G, BS_1, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_3, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_3[nl[n]]),
				&(boundevent1[nl[n]][23]), &(boundevent2[nl[n]][23]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec4_7[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][47]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][23],0);
					if (cond1) rc += MPI_Isend(&Buffersend2_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (23 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive4_7[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][47]);
					if (cond1) rc += MPI_Isend(&send2_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE], (23 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send2_4 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], BS_1 - N1G, BS_1, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_4, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_4[nl[n]]),
				&(boundevent1[nl[n]][24]), &(boundevent2[nl[n]][24]));
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec4_8[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][48]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][24],0);
					if (cond1) rc += MPI_Isend(&Buffersend2_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (24 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive4_8[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][48]);
					if (cond1) rc += MPI_Isend(&send2_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE], (24 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR2]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send to coarser grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send_average1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1 - (1 + REF_1) * N1G, BS_1, -N2G, BS_2 + N2G, -N3G, BS_3 + N3G,
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send2, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]),
					&(Buffersend2[nl[n]]), &(boundevent1[nl[n]][20]), &(boundevent2[nl[n]][20]));
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
							if (cond2) rc += MPI_Irecv(&Bufferrec4_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (45 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][45]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec4_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (46 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][46]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec4_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (47 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][47]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec4_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (48 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][48]);
						}
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][20],0);
						if (cond1) rc += MPI_Isend(&Buffersend2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
							if (cond2) rc += MPI_Irecv(&receive4_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (45 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][45]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&receive4_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (46 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][46]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive4_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (47 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][47]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&receive4_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (48 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][48]);
						}
						if (cond1) rc += MPI_Isend(&send2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR4]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[n][AMR_NBR4], 0, N1G, -N2G, BS_2 + N2G, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), send4, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4[nl[n]]),
				&(boundevent1[nl[n]][40]), &(boundevent2[nl[n]][40]));
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][20]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][40],0);
					if (cond1) rc += MPI_Isend(&Buffersend4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][20]);
					if (cond1) rc += MPI_Isend(&send4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR4]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send4_5 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, N1G, -N2G, BS_2 / (1 + REF_2) + N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send4_5, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_5[nl[n]]),
				&(boundevent1[nl[n]][45]), &(boundevent2[nl[n]][45]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec2_1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][21]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][45],0);
					if (cond1) rc += MPI_Isend(&Buffersend4_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (45 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive2_1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][21]);
					if (cond1) rc += MPI_Isend(&send4_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE], (45 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send4_6 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, N1G, -N2G, BS_2 / (1 + REF_2) + N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send4_6, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_6[nl[n]]),
				&(boundevent1[nl[n]][46]), &(boundevent2[nl[n]][46]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec2_2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][22]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][46],0);
					if (cond1) rc += MPI_Isend(&Buffersend4_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (46 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive2_2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][22]);
					if (cond1) rc += MPI_Isend(&send4_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE], (46 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send4_7 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send4_7, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_7[nl[n]]),
				&(boundevent1[nl[n]][47]), &(boundevent2[nl[n]][47]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec2_3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][23]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][47],0);
					if (cond1) rc += MPI_Isend(&Buffersend4_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (47 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive2_3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][23]);
					if (cond1) rc += MPI_Isend(&send4_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE], (47 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1 && REF_2 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send4_8 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] && block[n][AMR_NSTEP]%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
				(BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send4_8, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_8[nl[n]]),
				&(boundevent1[nl[n]][48]), &(boundevent2[nl[n]][48]));
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec2_4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][24]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][48],0);
					if (cond1) rc += MPI_Isend(&Buffersend4_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (48 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive2_4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (20 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][24]);
					if (cond1) rc += MPI_Isend(&send4_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE], (48 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR4]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send to coarser grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send_average1(n, block[block[n][AMR_NBR4]][AMR_PARENT], 0, (1 + REF_1)*N1G, -N2G, (BS_2 + N2G), -N3G, (BS_3 + N3G),
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send4, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4[nl[n]]),
					&(boundevent1[nl[n]][40]), &(boundevent2[nl[n]][40]));
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							if (cond2) rc += MPI_Irecv(&Bufferrec2_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (21 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][21]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec2_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (22 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][22]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec2_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (23 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][23]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec2_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (24 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][24]);
						}
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][40],0);
						if (cond1) rc += MPI_Isend(&Buffersend4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							if (cond2) rc += MPI_Irecv(&receive2_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (21 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][21]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&receive2_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (22 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][22]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive2_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (23 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][23]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
							if (cond2) rc += MPI_Irecv(&receive2_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (24 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][24]);
						}
						if (cond1) rc += MPI_Isend(&send4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE], (40 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
#endif
}

void bound_send2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep){
#if (MPI_enable)
	int cond1, cond2;
	//Exchange boundary cells for MPI threads
	//Positive X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR3]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR3]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send2(n, block[n][AMR_NBR3], -N1G, BS_1 + N1G, BS_2 - N2G, BS_2, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send3, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3[nl[n]]),
				&(boundevent1[nl[n]][30]), &(boundevent2[nl[n]][30]));
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
						if (cond2) rc += MPI_Irecv(&Bufferrec1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][10]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&Bufferrec1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][10]);
					}
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][30],0);
					if (cond1) rc += MPI_Isend(&Buffersend3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
						if (cond2) rc += MPI_Irecv(&receive1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][10]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][10]);
					}
					if (cond1) rc += MPI_Isend(&send3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send3_1 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], -N1G, BS_1 / (1 + REF_1) + N1G, BS_2 - N2G, BS_2, -N3G, BS_3 / (1 + REF_3) + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_1, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_1[nl[n]]),
					&(boundevent1[nl[n]][31]), &(boundevent2[nl[n]][31]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec1_3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][13]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][31],0);
						if (cond1) rc += MPI_Isend(&Buffersend3_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (31 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive1_3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][13]);
						if (cond1) rc += MPI_Isend(&send3_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE], (31 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send3_2 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], -N1G, BS_1 / (1 + REF_1) + N1G, BS_2 - N2G, BS_2, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_2, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_2[nl[n]]),
					&(boundevent1[nl[n]][32]), &(boundevent2[nl[n]][32]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec1_4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][14]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][32],0);
						if (cond1) rc += MPI_Isend(&Buffersend3_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (32 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive1_4[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][14]);
						if (cond1) rc += MPI_Isend(&send3_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE], (32 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send3_5 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, BS_2 - N2G, BS_2, -N3G, BS_3 / (1 + REF_3) + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_5, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_5[nl[n]]),
					&(boundevent1[nl[n]][35]), &(boundevent2[nl[n]][35]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec1_7[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][17]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][35],0);
						if (cond1) rc += MPI_Isend(&Buffersend3_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (35 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive1_7[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][17]);
						if (cond1) rc += MPI_Isend(&send3_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE], (35 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send3_6 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, BS_2 - N2G, BS_2, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_6, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_6[nl[n]]),
					&(boundevent1[nl[n]][36]), &(boundevent2[nl[n]][36]));
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec1_8[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][18]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][36],0);
						if (cond1) rc += MPI_Isend(&Buffersend3_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (36 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive1_8[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][18]);
						if (cond1) rc += MPI_Isend(&send3_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE], (36 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR3]][AMR_PARENT] >= 0){
				if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
					//send to coarser grid
					cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
					cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
					if (cond1) pack_send_average2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, (BS_1 + N1G), (BS_2 - (1 + REF_2) * N2G), BS_2, -N3G, (BS_3 + N3G),
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send3, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3[nl[n]]),
						&(boundevent1[nl[n]][30]), &(boundevent2[nl[n]][30]));
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
							if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
								if (cond2) rc += MPI_Irecv(&Bufferrec1_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (13 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][13]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec1_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (14 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][14]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec1_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (17 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][17]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec1_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (18 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][18]);
							}
							if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][30],0);
							if (cond1) rc += MPI_Isend(&Buffersend3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						}
						else{
							if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
								if (cond2) rc += MPI_Irecv(&receive1_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (13 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][13]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&receive1_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (14 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][14]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
								if (cond2) rc += MPI_Irecv(&receive1_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (17 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][17]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&receive1_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (18 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][18]);
							}
							if (cond1) rc += MPI_Isend(&send3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						}
						if (cond1) MPI_Request_free(&req[nl[n]]);
					}
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR1]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR1]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send2(n, block[n][AMR_NBR1], -N1G, BS_1 + N1G, 0, N2G, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send1, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1[nl[n]]),
				&(boundevent1[nl[n]][10]), &(boundevent2[nl[n]][10]));
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
						if (cond2) rc += MPI_Irecv(&Bufferrec3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][30]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&Bufferrec3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][30]);
					}
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][10],0);
					if (cond1) rc += MPI_Isend(&Buffersend1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
						if (cond2) rc += MPI_Irecv(&receive3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][30]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive3[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][30]);
					}
					if (cond1) rc += MPI_Isend(&send1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G)*(BS_1 + 2 * N1G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR1]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2){
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send1_3 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], -N1G, BS_1 / (1 + REF_1) + N1G, 0, N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_3, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_3[nl[n]]),
					&(boundevent1[nl[n]][13]), &(boundevent2[nl[n]][13]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec3_1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][31]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][13],0);
						if (cond1) rc += MPI_Isend(&Buffersend1_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (13 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive3_1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][31]);
						if (cond1) rc += MPI_Isend(&send1_3[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE], (13 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send1_4 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], -N1G, BS_1 / (1 + REF_1) + N1G, 0, N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_4, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_4[nl[n]]),
					&(boundevent1[nl[n]][14]), &(boundevent2[nl[n]][14]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec3_2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][32]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][14],0);
						if (cond1) rc += MPI_Isend(&Buffersend1_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (14 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive3_2[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][32]);
						if (cond1) rc += MPI_Isend(&send1_4[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE], (14 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send1_7 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, 0, N2G, -N3G, BS_3 / (1 + REF_3) + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_7, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_7[nl[n]]),
					&(boundevent1[nl[n]][17]), &(boundevent2[nl[n]][17]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec3_5[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][35]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][17],0);
						if (cond1) rc += MPI_Isend(&Buffersend1_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (17 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive3_5[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][35]);
						if (cond1) rc += MPI_Isend(&send1_7[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE], (17 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1 && REF_1 == 1 && REF_3 == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send1_8 to finer grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, 0, N2G, BS_3 / (1 + REF_3) - N3G, BS_3 + N3G,
					(BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_8, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_8[nl[n]]),
					&(boundevent1[nl[n]][18]), &(boundevent2[nl[n]][18]));
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (cond2) rc += MPI_Irecv(&Bufferrec3_6[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][36]);
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][18],0);
						if (cond1) rc += MPI_Isend(&Buffersend1_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (18 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (cond2) rc += MPI_Irecv(&receive3_6[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (30 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][36]);
						if (cond1) rc += MPI_Isend(&send1_8[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE], (18 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
			if (block[block[n][AMR_NBR1]][AMR_PARENT] >= 0){
				if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
					//send to coarser grid
					cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
					cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
					if (cond1) pack_send_average2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, 0, (1 + REF_2) * N2G, -N3G, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send1, prim, ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1[nl[n]]),
						&(boundevent1[nl[n]][10]), &(boundevent2[nl[n]][10]));
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (gpu == 1){
							if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
								if (cond2) rc += MPI_Irecv(&Bufferrec3_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (31 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][31]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec3_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (32 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][32]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec3_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (35 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][35]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&Bufferrec3_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (36 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][36]);
							}
							if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][10],0);
							if (cond1) rc += MPI_Isend(&Buffersend1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						}
						else{
							if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
								if (cond2) rc += MPI_Irecv(&receive3_1[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (31 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][31]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&receive3_2[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (32 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][32]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
								if (cond2) rc += MPI_Irecv(&receive3_5[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (35 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][35]);
							}
							if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
								if (cond2) rc += MPI_Irecv(&receive3_6[nl[n]][0], (NPR + 3)*(BS_3 / (1 + REF_3) + 2 * N3G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (36 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][36]);
							}
							if (cond1) rc += MPI_Isend(&send1[nl[n]][0], (NPR + 3)*(BS_3 + 2 * N3G) / (1 + REF_3)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE], (10 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
						}
						if (cond1) MPI_Request_free(&req[nl[n]]);
					}
				}
			}
		}
	}
#endif
}

void bound_send3(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep){
#if (MPI_enable)
	int cond1, cond2;
	//Positive X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR5]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[n][AMR_NBR5], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G, BS_3 - N3G, BS_3, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), send5, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5[nl[n]]),
				&(boundevent1[nl[n]][50]), &(boundevent2[nl[n]][50]));
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][60]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][50],0);
					if (cond1) rc += MPI_Isend(&Buffersend5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][60]);
					if (cond1) rc += MPI_Isend(&send5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR5]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send5_1 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], -N1G, BS_1 / (1 + REF_1) + N1G, -N2G, BS_2 / (1 + REF_2) + N2G, BS_3 - N3G, BS_3,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_1, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_1[nl[n]]),
				&(boundevent1[nl[n]][51]), &(boundevent2[nl[n]][51]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec6_2[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][62]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][51],0);
					if (cond1) rc += MPI_Isend(&Buffersend5_1[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (51 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive6_2[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][62]);
					if (cond1) rc += MPI_Isend(&send5_1[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE], (51 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send5_3 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], -N1G, BS_1 / (1 + REF_1) + N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, BS_3 - N3G, BS_3,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_3, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_3[nl[n]]),
				&(boundevent1[nl[n]][53]), &(boundevent2[nl[n]][53]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec6_4[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][64]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][53],0);
					if (cond1) rc += MPI_Isend(&Buffersend5_3[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (53 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive6_4[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][64]);
					if (cond1) rc += MPI_Isend(&send5_3[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE], (53 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send5_5 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, -N2G, BS_2 / (1 + REF_2) + N2G, BS_3 - N3G, BS_3,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_5, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_5[nl[n]]),
				&(boundevent1[nl[n]][55]), &(boundevent2[nl[n]][55]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec6_6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][66]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][55],0);
					if (cond1) rc += MPI_Isend(&Buffersend5_5[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (55 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive6_6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][66]);
					if (cond1) rc += MPI_Isend(&send5_5[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE], (55 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send5_7 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, BS_3 - N3G, BS_3,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_7, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_7[nl[n]]),
				&(boundevent1[nl[n]][57]), &(boundevent2[nl[n]][57]));
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec6_8[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][68]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][57],0);
					if (cond1) rc += MPI_Isend(&Buffersend5_7[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (57 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive6_8[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][68]);
					if (cond1) rc += MPI_Isend(&send5_7[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE], (57 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR5]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send to coarser grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send_average3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, (BS_1 + N1G), -N2G, (BS_2 + N2G), BS_3 - (1 + REF_3) * N3G, BS_3,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send5, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5[nl[n]]),
					&(boundevent1[nl[n]][50]), &(boundevent2[nl[n]][50]));
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){

					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
							if (cond2) rc += MPI_Irecv(&Bufferrec6_2[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (62 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][62]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec6_4[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (64 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][64]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec6_6[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (66 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][66]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec6_8[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (68 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][68]);
						}
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][50],0);
						if (cond1) rc += MPI_Isend(&Buffersend5[nl[n]][0], (NPR + 3)*(BS_2 + 2 * N2G) / (1 + REF_2)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
							if (cond2) rc += MPI_Irecv(&receive6_2[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (62 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][62]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive6_4[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (64 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][64]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
							if (cond2) rc += MPI_Irecv(&receive6_6[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (66 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][66]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive6_8[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (68 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][68]);
						}
						if (cond1) rc += MPI_Isend(&send5[nl[n]][0], (NPR + 3)*(BS_2 + 2 * N2G) / (1 + REF_2)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}

	//Negative X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1 && (nstep%block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[n][AMR_NBR6]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[n][AMR_NBR6], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G, 0, N3G, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), send6, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6[nl[n]]),
				&(boundevent1[nl[n]][60]), &(boundevent2[nl[n]][60]));
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][50]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][60],0);
					if (cond1) rc += MPI_Isend(&Buffersend6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[n][AMR_NBR6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][50]);
					if (cond1) rc += MPI_Isend(&send6[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG, MPI_DOUBLE, block[block[n][AMR_NBR6]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send6_2 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], -N1G, BS_1 / (1 + REF_1) + N1G, -N2G, BS_2 / (1 + REF_2) + N2G, 0, N3G,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_2, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_2[nl[n]]),
				&(boundevent1[nl[n]][62]), &(boundevent2[nl[n]][62]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec5_1[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][51]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][62],0);
					if (cond1) rc += MPI_Isend(&Buffersend6_2[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (62 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive5_1[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][51]);
					if (cond1) rc += MPI_Isend(&send6_2[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE], (62 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send6_4 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], -N1G, BS_1 / (1 + REF_1) + N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, 0, N3G,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_4, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_4[nl[n]]),
				&(boundevent1[nl[n]][64]), &(boundevent2[nl[n]][64]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec5_3[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][53]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][64],0);
					if (cond1) rc += MPI_Isend(&Buffersend6_4[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (64 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive5_3[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][53]);
					if (cond1) rc += MPI_Isend(&send6_4[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE], (64 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send6_6 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, -N2G, BS_2 / (1 + REF_2) + N2G, 0, N3G,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_6, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_6[nl[n]]),
				&(boundevent1[nl[n]][66]), &(boundevent2[nl[n]][66]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec5_5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][55]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][66],0);
					if (cond1) rc += MPI_Isend(&Buffersend6_6[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (66 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive5_5[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][55]);
					if (cond1) rc += MPI_Isend(&send6_6[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE], (66 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1 && REF_1 == 1 && REF_2 == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
			//send6_8 to finer grid
			cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
			cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
			if (cond1) pack_send3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], BS_1 / (1 + REF_1) - N1G, BS_1 + N1G, BS_2 / (1 + REF_2) - N2G, BS_2 + N2G, 0, N3G,
				(BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_8, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_8[nl[n]]),
				&(boundevent1[nl[n]][68]), &(boundevent2[nl[n]][68]));
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
				if (gpu == 1){
					if (cond2) rc += MPI_Irecv(&Bufferrec5_7[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][57]);
					if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][68],0);
					if (cond1) rc += MPI_Isend(&Buffersend6_8[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (68 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				else{
					if (cond2) rc += MPI_Irecv(&receive5_7[nl[n]][0], (NPR + 3)*(BS_1 + 2 * N1G) / (1 + REF_1)*(BS_2 + 2 * N2G) / (1 + REF_2) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (50 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][57]);
					if (cond1) rc += MPI_Isend(&send6_8[nl[n]][0], (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE], (68 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
				}
				if (cond1) MPI_Request_free(&req[nl[n]]);
			}
		}
		if (block[block[n][AMR_NBR6]][AMR_PARENT] >= 0){
			if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1 && (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || prestep == 1)){
				//send to coarser grid
				cond1 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] > block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[n][AMR_TIMELEVEL] == 0));
				cond2 = (prestep == 0 || (prestep == 1 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && block[n][AMR_NSTEP] % block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1));
				if (cond1) pack_send_average3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, (BS_1 + N1G), -N2G, (BS_2 + N2G), 0, (1 + REF_3) * N3G,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send6, prim,ps, &(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6[nl[n]]),
					&(boundevent1[nl[n]][60]), &(boundevent2[nl[n]][60]));
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (gpu == 1){
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							if (cond2) rc += MPI_Irecv(&Bufferrec5_1[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (51 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][51]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec5_3[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (53 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][53]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec5_5[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (55 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][55]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&Bufferrec5_7[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (57 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][57]);
						}
						if (cond1) cudaStreamSynchronize(commandQueueGPU[nl[n]]); //cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][60],0);
						if (cond1) rc += MPI_Isend(&Buffersend6[nl[n]][0], (NPR + 3)*(BS_2 + 2 * N2G) / (1 + REF_2)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					else{
						if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
							if (cond2) rc += MPI_Irecv(&receive5_1[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (51 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][51]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive5_3[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (53 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][53]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
							if (cond2) rc += MPI_Irecv(&receive5_5[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (55 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][55]);
						}
						if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
							if (cond2) rc += MPI_Irecv(&receive5_7[nl[n]][0], (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_1 / (1 + REF_1) + 2 * N1G) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (57 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n]][57]);
						}
						if (cond1) rc += MPI_Isend(&send6[nl[n]][0], (NPR + 3)*(BS_2 + 2 * N2G) / (1 + REF_2)*(BS_1 + 2 * N1G) / (1 + REF_1) * NG, MPI_DOUBLE, block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE], (60 * n_active_total + 50 * n_active_total + 70 * n_active_total * (prestep==1) + block[n][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[nl[n]]);
					}
					if (cond1) MPI_Request_free(&req[nl[n]]);
				}
			}
		}
	}
	//MPI_Barrier(mpi_cartcomm);
#endif
}

/*Receive boundaries for compute nodes through MPI*/
void bound_rec1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n){
#if (MPI_enable)
	//positive X1
	if (block[n][AMR_NBR4] >= 0){
		if (block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR4]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[block[n][AMR_NBR4]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 &&  block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][20], &Statbound[nl[n]][20]);
				}
				unpack_receive1(n, block[n][AMR_NBR4], 0, -N1G, 0, 0, -N2G, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), receive2, tempreceive2, prim,
					&(Bufferp[nl[n]]), &(Bufferrec2[nl[n]]), &(tempBufferrec2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[n][AMR_NBR4], 0, -N1G, 0, 0, -N2G, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), send2, tempreceive2, prim,
					&(Bufferp[nl[n]]), &(Buffersend2[nl[block[n][AMR_NBR4]]]), &(tempBufferrec2[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR4]]][20]), NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR4]][AMR_TIMELEVEL]);
			}
		}

		else if (block[block[n][AMR_NBR4]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][21], &Statbound[nl[n]][21]);
				}
				unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, -N1G, 0, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive2_1, tempreceive2_1, prim,
					&(Bufferp[nl[n]]), &(Bufferrec2_1[nl[n]]), &(tempBufferrec2_1[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD5], 0, -N1G, 0, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send2, tempreceive2_1, prim,
					&(Bufferp[nl[n]]), &(Buffersend2[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]]), &(tempBufferrec2_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_CHILD5]]][20]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD5]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][22], &Statbound[nl[n]][22]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, -N1G, 0, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive2_2, tempreceive2_2, prim,
						&(Bufferp[nl[n]]), &(Bufferrec2_2[nl[n]]), &(tempBufferrec2_2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD6], 0, -N1G, 0, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send2, tempreceive2_2, prim,
						&(Bufferp[nl[n]]), &(Buffersend2[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]]), &(tempBufferrec2_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_CHILD6]]][20]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 &&  block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][23], &Statbound[nl[n]][23]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, -N1G, 0, 1, BS_2 / (1 + REF_2), BS_2 + REF_2*D2, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive2_3, tempreceive2_3, prim,
						&(Bufferp[nl[n]]), &(Bufferrec2_3[nl[n]]), &(tempBufferrec2_3[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD7], 0, -N1G, 0, 1, BS_2 / (1 + REF_2), BS_2 + REF_2*D2, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send2, tempreceive2_3, prim,
						&(Bufferp[nl[n]]), &(Buffersend2[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]]), &(tempBufferrec2_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_CHILD7]]][20]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][24], &Statbound[nl[n]][24]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, -N1G, 0, 1, BS_2 / (1 + REF_2), BS_2 + REF_2*D2, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive2_4, tempreceive2_4, prim,
						&(Bufferp[nl[n]]), &(Bufferrec2_4[nl[n]]), &(tempBufferrec2_4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR4]][AMR_CHILD8], 0, -N1G, 0, 1, BS_2 / (1 + REF_2), BS_2 + REF_2*D2, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send2, tempreceive2_4, prim,
						&(Bufferp[nl[n]]), &(Buffersend2[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]]), &(tempBufferrec2_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_CHILD8]]][20]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][21], &Statbound[nl[n]][21]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), receive2_1, send4_5, tempreceive2_1, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec2_1[nl[n]]), &(Buffersend4_5[nl[n]]), &(tempBufferrec2_1[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_1, receive2_1, tempreceive2_1, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_1[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]]), &(Bufferrec2_1[nl[n]]), &(tempBufferrec2_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]][21]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][22], &Statbound[nl[n]][22]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), receive2_2, send4_6, tempreceive2_2, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec2_2[nl[n]]), &(Buffersend4_6[nl[n]]), &(tempBufferrec2_2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_2, receive2_2, tempreceive2_2, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_2[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]]), &(Bufferrec2_2[nl[n]]), &(tempBufferrec2_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]][22]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][23], &Statbound[nl[n]][23]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), receive2_3, send4_7, tempreceive2_3, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec2_3[nl[n]]), &(Buffersend4_7[nl[n]]), &(tempBufferrec2_3[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_3, receive2_3, tempreceive2_3, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_3[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]]), &(Bufferrec2_3[nl[n]]), &(tempBufferrec2_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]][23]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][24], &Statbound[nl[n]][24]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), receive2_4, send4_8, tempreceive2_4, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec2_4[nl[n]]), &(Buffersend4_8[nl[n]]), &(tempBufferrec2_4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR4]][AMR_PARENT], -N1G, 0, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, (BS_2 / (1 + REF_2) + 2 * N2G), (BS_3 / (1 + REF_3) + 2 * N3G), send2_4, receive2_4, tempreceive2_4, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend2_4[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]]), &(Bufferrec2_4[nl[n]]), &(tempBufferrec2_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR4]][AMR_PARENT]]][24]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR4]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else fprintf(stderr, "Error in indexing!\n");
	}

	//Negative X1
	if (block[n][AMR_NBR2] >= 0){
		if (block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1){
			//receive from same level grid
			if (block[block[n][AMR_NBR2]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[block[n][AMR_NBR2]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][40], &Statbound[nl[n]][40]);
				}
				unpack_receive1(n, block[n][AMR_NBR2], 0, BS_1, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), receive4, tempreceive4, prim,
					&(Bufferp[nl[n]]), &(Bufferrec4[nl[n]]), &(tempBufferrec4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[n][AMR_NBR2], 0, BS_1, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_2 + 2 * N2G), (BS_3 + 2 * N3G), send4, tempreceive4, prim,
					&(Bufferp[nl[n]]), &(Buffersend4[nl[block[n][AMR_NBR2]]]), &(tempBufferrec4[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR2]]][40]), NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR2]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR2]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][45], &Statbound[nl[n]][45]);
				}
				unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], 0, BS_1, BS_1 + N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive4_5, tempreceive4_5, prim,
					&(Bufferp[nl[n]]), &(Bufferrec4_5[nl[n]]), &(tempBufferrec4_5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD1], 0, BS_1, BS_1 + N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
					(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send4, tempreceive4_5, prim,
					&(Bufferp[nl[n]]), &(Buffersend4[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]]), &(tempBufferrec4_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_CHILD1]]][40]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			if (REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][46], &Statbound[nl[n]][46]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], 0, BS_1, BS_1 + N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive4_6, tempreceive4_6, prim,
						&(Bufferp[nl[n]]), &(Bufferrec4_6[nl[n]]), &(tempBufferrec4_6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD2], 0, BS_1, BS_1 + N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 1, BS_3 / (1 + REF_3), BS_3 + REF_3*D3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send4, tempreceive4_6, prim,
						&(Bufferp[nl[n]]), &(Buffersend4[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]]), &(tempBufferrec4_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_CHILD2]]][40]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][47], &Statbound[nl[n]][47]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], 0, BS_1, BS_1 + N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive4_7, tempreceive4_7, prim,
						&(Bufferp[nl[n]]), &(Bufferrec4_7[nl[n]]), &(tempBufferrec4_7[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD3], 0, BS_1, BS_1 + N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send4, tempreceive4_7, prim,
						&(Bufferp[nl[n]]), &(Buffersend4[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]]), &(tempBufferrec4_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_CHILD3]]][40]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
			}
			if (REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][48], &Statbound[nl[n]][48]);
					}
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], 0, BS_1, BS_1 + N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), receive4_8, tempreceive4_8, prim,
						&(Bufferp[nl[n]]), &(Bufferrec4_8[nl[n]]), &(tempBufferrec4_8[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive1(n, block[block[n][AMR_NBR2]][AMR_CHILD4], 0, BS_1, BS_1 + N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
						(BS_2 + 2 * N2G) / (1 + REF_2), (BS_3 + 2 * N3G) / (1 + REF_3), send4, tempreceive4_8, prim,
						&(Bufferp[nl[n]]), &(Buffersend4[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]]), &(tempBufferrec4_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_CHILD4]]][40]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
						MPI_Wait(&boundreqs[nl[n]][45], &Statbound[nl[n]][45]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), receive4_5, send2_1, tempreceive4_5, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec4_5[nl[n]]), &(Buffersend2_1[nl[n]]), &(tempBufferrec4_5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), send4_5, receive4_5, tempreceive4_5, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_5[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]]), &(Bufferrec4_5[nl[n]]), &(tempBufferrec4_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]][45]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
						MPI_Wait(&boundreqs[nl[n]][46], &Statbound[nl[n]][46]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), receive4_6, send2_2, tempreceive4_6, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec4_6[nl[n]]), &(Buffersend2_2[nl[n]]), &(tempBufferrec4_6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), send4_6, receive4_6, tempreceive4_6, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_6[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]]), &(Bufferrec4_6[nl[n]]), &(tempBufferrec4_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]][46]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
						MPI_Wait(&boundreqs[nl[n]][47], &Statbound[nl[n]][47]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), receive4_7, send2_3, tempreceive4_7, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec4_7[nl[n]]), &(Buffersend2_3[nl[n]]), &(tempBufferrec4_7[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), send4_7, receive4_7, tempreceive4_7, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_7[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]]), &(Bufferrec4_7[nl[n]]), &(tempBufferrec4_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]][47]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_2 == 1 && REF_3 == 1){
				if (block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if ((nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
						MPI_Wait(&boundreqs[nl[n]][48], &Statbound[nl[n]][48]);
					}
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), receive4_8, send2_4, tempreceive4_8, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec4_8[nl[n]]), &(Buffersend2_4[nl[n]]), &(tempBufferrec4_8[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse1(n, block[block[n][AMR_NBR2]][AMR_PARENT], BS_1, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, BS_3 + N3G, BS_2 / (1 + REF_2) + 2 * N2G, (BS_3 / (1 + REF_3) + 2 * N3G), send4_8, receive4_8, tempreceive4_8, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend4_8[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]]), &(Bufferrec4_8[nl[n]]), &(tempBufferrec4_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR2]][AMR_PARENT]]][48]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR2]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else fprintf(stderr, "Error in indexing!\n");
	}
#endif
}

void bound_rec2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n){
#if (MPI_enable)
	//Positive X2
	if (block[n][AMR_NBR1] >= 0){
		if (block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[block[n][AMR_NBR1]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR1]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][30], &Statbound[nl[n]][30]);
				}
				if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, BS_1 + N1G, 0, -N2G, 0, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), receive3, tempreceive3, prim,
						&(Bufferp[nl[n]]), &(Bufferrec3[nl[n]]), &(tempBufferrec3[nl[n]]), NULL, NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, BS_1 + N1G, 0, -N2G, 0, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), receive3, tempreceive3, prim,
						&(Bufferp[nl[n]]), &(Bufferrec3[nl[n]]), &(tempBufferrec3[nl[n]]), NULL, NULL, 0);
				}
			}
			else{
				if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, BS_1 + N1G, 0, -N2G, 0, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send1, tempreceive3, prim,
						&(Bufferp[nl[n]]), &(Buffersend1[nl[block[n][AMR_NBR1]]]), &(tempBufferrec3[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR1]]][10]), NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR1], 0, -N1G, BS_1 + N1G, 0, -N2G, 0, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send3, tempreceive3, prim,
						&(Bufferp[nl[n]]), &(Buffersend3[nl[block[n][AMR_NBR1]]]), &(tempBufferrec3[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR1]]][30]), NULL, 0);
				}
			}
		}
		else if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2){
			if (block[block[n][AMR_NBR1]][AMR_REFINED] == 1){
				//receive from finer grid
				if (block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][31], &Statbound[nl[n]][31]);
					}
					unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive3_1, tempreceive3_1, prim,
						&(Bufferp[nl[n]]), &(Bufferrec3_1[nl[n]]), &(tempBufferrec3_1[nl[n]]), NULL, NULL, 0);
				}
				else{
					unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD3], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send3, tempreceive3_1, prim,
						&(Bufferp[nl[n]]), &(Buffersend3[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]]), &(tempBufferrec3_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_CHILD3]]][30]), NULL, 0);
				}
				if (REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][32], &Statbound[nl[n]][32]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive3_2, tempreceive3_2, prim,
							&(Bufferp[nl[n]]), &(Bufferrec3_2[nl[n]]), &(tempBufferrec3_2[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD4], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G, 0, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send3, tempreceive3_2, prim,
							&(Bufferp[nl[n]]), &(Buffersend3[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]]), &(tempBufferrec3_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_CHILD4]]][30]), NULL, 0);
					}
				}
				if (REF_1 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][35], &Statbound[nl[n]][35]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G, 0, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive3_5, tempreceive3_5, prim,
							&(Bufferp[nl[n]]), &(Bufferrec3_5[nl[n]]), &(tempBufferrec3_5[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD7], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G, 0, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send3, tempreceive3_5, prim,
							&(Bufferp[nl[n]]), &(Buffersend3[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]]), &(tempBufferrec3_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_CHILD7]]][30]), NULL, 0);
					}
				}
				if (REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][36], &Statbound[nl[n]][36]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G, 0, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive3_6, tempreceive3_6, prim,
							&(Bufferp[nl[n]]), &(Bufferrec3_6[nl[n]]), &(tempBufferrec3_6[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR1]][AMR_CHILD8], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G, 0, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send3, tempreceive3_6, prim,
							&(Bufferp[nl[n]]), &(Buffersend3[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]]), &(tempBufferrec3_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_CHILD8]]][30]), NULL, 0);
					}
				}
			}
			else if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//receive from coarser grid
				if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
							MPI_Wait(&boundreqs[nl[n]][31], &Statbound[nl[n]][31]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive3_1, send1_3, tempreceive3_1, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec3_1[nl[n]]), &(Buffersend1_3[nl[n]]), &(tempBufferrec3_1[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_1, receive3_1, tempreceive3_1, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_1[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]]), &(Bufferrec3_1[nl[n]]), &(tempBufferrec3_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]][31]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
							MPI_Wait(&boundreqs[nl[n]][32], &Statbound[nl[n]][32]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive3_2, send1_4, tempreceive3_2, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec3_2[nl[n]]), &(Buffersend1_4[nl[n]]), &(tempBufferrec3_2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_2, receive3_2, tempreceive3_2, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_2[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]]), &(Bufferrec3_2[nl[n]]), &(tempBufferrec3_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]][32]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
							MPI_Wait(&boundreqs[nl[n]][35], &Statbound[nl[n]][35]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive3_5, send1_7, tempreceive3_5, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec3_5[nl[n]]), &(Buffersend1_7[nl[n]]), &(tempBufferrec3_5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_5, receive3_5, tempreceive3_5, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_5[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]]), &(Bufferrec3_5[nl[n]]), &(tempBufferrec3_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]][35]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if ((nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1))){
							MPI_Wait(&boundreqs[nl[n]][36], &Statbound[nl[n]][36]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive3_6, send1_8, tempreceive3_6, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec3_6[nl[n]]), &(Buffersend1_8[nl[n]]), &(tempBufferrec3_6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR1]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, 0,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send3_6, receive3_6, tempreceive3_6, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend3_6[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]]), &(Bufferrec3_6[nl[n]]), &(tempBufferrec3_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR1]][AMR_PARENT]]][36]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR1]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
			}
		}
	}

	//Negative X2
	if (block[n][AMR_NBR3] >= 0){
		if (block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[block[n][AMR_NBR3]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR3]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][10], &Statbound[nl[n]][10]);
				}
				if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, BS_1 + N1G, 0, BS_2, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), receive1, tempreceive1, prim,
						&(Bufferp[nl[n]]), &(Bufferrec1[nl[n]]), &(tempBufferrec1[nl[n]]), NULL, NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, BS_1 + N1G, 0, BS_2, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), receive1, tempreceive1, prim,
						&(Bufferp[nl[n]]), &(Bufferrec1[nl[n]]), &(tempBufferrec1[nl[n]]), NULL, NULL, 0);
				}
			}
			else{
				if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, BS_1 + N1G, 0, BS_2, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send3, tempreceive1, prim,
						&(Bufferp[nl[n]]), &(Buffersend3[nl[block[n][AMR_NBR3]]]), &(tempBufferrec1[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR3]]][30]), NULL, 1);
				}
				else{
					unpack_receive2(n, block[n][AMR_NBR3], 0, -N1G, BS_1 + N1G, 0, BS_2, BS_2 + N2G, 0, -N3G, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_3 + 2 * N3G), send1, tempreceive1, prim,
						&(Bufferp[nl[n]]), &(Buffersend1[nl[block[n][AMR_NBR3]]]), &(tempBufferrec1[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR3]]][10]), NULL, 0);
				}
			}
		}
		else if (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1){
			if (block[block[n][AMR_NBR3]][AMR_REFINED] == 1){
				//receive from finer grid
				if (block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][13], &Statbound[nl[n]][13]);
					}
					unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, BS_2, BS_2 + N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive1_3, tempreceive1_3, prim,
						&(Bufferp[nl[n]]), &(Bufferrec1_3[nl[n]]), &(tempBufferrec1_3[nl[n]]), NULL, NULL, 0);
				}
				else{
					unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD1], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, BS_2, BS_2 + N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send1, tempreceive1_3, prim,
						&(Bufferp[nl[n]]), &(Buffersend1[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]]), &(tempBufferrec1_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_CHILD1]]][10]), NULL, 0);
				}
				if (REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][14], &Statbound[nl[n]][14]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, BS_2, BS_2 + N2G, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive1_4, tempreceive1_4, prim,
							&(Bufferp[nl[n]]), &(Bufferrec1_4[nl[n]]), &(tempBufferrec1_4[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD2], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, BS_2, BS_2 + N2G, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send1, tempreceive1_4, prim,
							&(Bufferp[nl[n]]), &(Buffersend1[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]]), &(tempBufferrec1_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_CHILD2]]][10]), NULL, 0);
					}
				}
				if (REF_1 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][17], &Statbound[nl[n]][17]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, BS_2, BS_2 + N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive1_7, tempreceive1_7, prim,
							&(Bufferp[nl[n]]), &(Bufferrec1_7[nl[n]]), &(tempBufferrec1_7[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD5], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, BS_2, BS_2 + N2G, 0, -N3G / (1 + REF_3), BS_3 / (1 + REF_3) + (1 - REF_3)*N3G,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send1, tempreceive1_7, prim,
							&(Bufferp[nl[n]]), &(Buffersend1[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]]), &(tempBufferrec1_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_CHILD5]]][10]), NULL, 0);
					}
				}
				if (REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][18], &Statbound[nl[n]][18]);
						}
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, BS_2, BS_2 + N2G, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), receive1_8, tempreceive1_8, prim,
							&(Bufferp[nl[n]]), &(Bufferrec1_8[nl[n]]), &(tempBufferrec1_8[nl[n]]), NULL, NULL, 0);
					}
					else{
						unpack_receive2(n, block[block[n][AMR_NBR3]][AMR_CHILD6], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, BS_2, BS_2 + N2G, 1, BS_3 / (1 + REF_3), BS_3 + D3*REF_3,
							(BS_1 + 2 * N1G) / (1 + REF_1), (BS_3 + 2 * N3G) / (1 + REF_3), send1, tempreceive1_8, prim,
							&(Bufferp[nl[n]]), &(Buffersend1[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]]), &(tempBufferrec1_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_CHILD6]]][10]), NULL, 0);
					}
				}
			}
			else if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1){
				//receive from coarser grid
				if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][13], &Statbound[nl[n]][13]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive1_3, send3_1, tempreceive1_3, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec1_3[nl[n]]), &(Buffersend3_1[nl[n]]), &(tempBufferrec1_3[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_3, receive1_3, tempreceive1_3, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_3[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]]), &(Bufferrec1_3[nl[n]]), &(tempBufferrec1_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]][13]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][14], &Statbound[nl[n]][14]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive1_4, send3_2, tempreceive1_4, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec1_4[nl[n]]), &(Buffersend3_2[nl[n]]), &(tempBufferrec1_4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_4, receive1_4, tempreceive1_4, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_4[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]]), &(Bufferrec1_4[nl[n]]), &(tempBufferrec1_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]][14]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][17], &Statbound[nl[n]][17]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive1_7, send3_5, tempreceive1_7, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec1_7[nl[n]]), &(Buffersend3_5[nl[n]]), &(tempBufferrec1_7[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_7, receive1_7, tempreceive1_7, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_7[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]]), &(Bufferrec1_7[nl[n]]), &(tempBufferrec1_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]][17]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
				if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_3 == 1){
					if (block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
						if (nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
							MPI_Wait(&boundreqs[nl[n]][18], &Statbound[nl[n]][18]);
						}
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), receive1_8, send3_6, tempreceive1_8, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec1_8[nl[n]]), &(Buffersend3_6[nl[n]]), &(tempBufferrec1_8[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
					else{
						unpack_receive_coarse2(n, block[block[n][AMR_NBR3]][AMR_PARENT], -N1G, BS_1 + N1G, BS_2, BS_2 + N2G,
							-N3G, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_3 / (1 + REF_3) + 2 * N3G), send1_8, receive1_8, tempreceive1_8, prim, ps,
							&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend1_8[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]]), &(Bufferrec1_8[nl[n]]), &(tempBufferrec1_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR3]][AMR_PARENT]]][18]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR3]][AMR_PARENT]][AMR_TIMELEVEL]);
					}
				}
			}
		}
	}
#endif
}

void bound_rec3(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n){
#if (MPI_enable)
	//Positive X3
	if (block[n][AMR_NBR6] >= 0){
		if (block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR6]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[block[n][AMR_NBR6]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][50], &Statbound[nl[n]][50]);
				}
				unpack_receive3(n, block[n][AMR_NBR6], 0, -N1G, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, -N3G, 0, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), receive5, tempreceive5, prim,
					&(Bufferp[nl[n]]), &(Bufferrec5[nl[n]]), &(tempBufferrec5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[n][AMR_NBR6], 0, -N1G, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, -N3G, 0, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), send5, tempreceive5, prim,
					&(Bufferp[nl[n]]), &(Buffersend5[nl[block[n][AMR_NBR6]]]), &(tempBufferrec5[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR6]]][50]), NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR6]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR6]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][51], &Statbound[nl[n]][51]);
				}
				unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive5_1, tempreceive5_1, prim,
					&(Bufferp[nl[n]]), &(Bufferrec5_1[nl[n]]), &(tempBufferrec5_1[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD2], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send5, tempreceive5_1, prim,
					&(Bufferp[nl[n]]), &(Buffersend5[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]]), &(tempBufferrec5_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_CHILD2]]][50]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD2]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][53], &Statbound[nl[n]][53]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive5_3, tempreceive5_3, prim,
						&(Bufferp[nl[n]]), &(Bufferrec5_3[nl[n]]), &(tempBufferrec5_3[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD4], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send5, tempreceive5_3, prim,
						&(Bufferp[nl[n]]), &(Buffersend5[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]]), &(tempBufferrec5_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_CHILD4]]][50]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD4]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][55], &Statbound[nl[n]][55]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive5_5, tempreceive5_5, prim,
						&(Bufferp[nl[n]]), &(Bufferrec5_5[nl[n]]), &(tempBufferrec5_5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD6], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send5, tempreceive5_5, prim,
						&(Bufferp[nl[n]]), &(Buffersend5[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]]), &(tempBufferrec5_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_CHILD6]]][50]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][57], &Statbound[nl[n]][57]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive5_7, tempreceive5_7, prim,
						&(Bufferp[nl[n]]), &(Bufferrec5_7[nl[n]]), &(tempBufferrec5_7[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR6]][AMR_CHILD8], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, -N3G, 0,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send5, tempreceive5_7, prim,
						&(Bufferp[nl[n]]), &(Buffersend5[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]]), &(tempBufferrec5_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_CHILD8]]][50]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD1] == n){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][51], &Statbound[nl[n]][51]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive5_1, send6_2, tempreceive5_1, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec5_1[nl[n]]), &(Buffersend6_2[nl[n]]), &(tempBufferrec5_1[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_1, receive5_1, tempreceive5_1, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_1[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]]), &(Bufferrec5_1[nl[n]]), &(tempBufferrec5_1[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]][51]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD3] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][53], &Statbound[nl[n]][53]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive5_3, send6_4, tempreceive5_3, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec5_3[nl[n]]), &(Buffersend6_4[nl[n]]), &(tempBufferrec5_3[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_3, receive5_3, tempreceive5_3, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_3[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]]), &(Bufferrec5_3[nl[n]]), &(tempBufferrec5_3[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]][53]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD5] == n && REF_1 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][55], &Statbound[nl[n]][55]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive5_5, send6_6, tempreceive5_5, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec5_5[nl[n]]), &(Buffersend6_6[nl[n]]), &(tempBufferrec5_5[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_5, receive5_5, tempreceive5_5, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_5[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]]), &(Bufferrec5_5[nl[n]]), &(tempBufferrec5_5[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]][55]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD7] == n && REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][57], &Statbound[nl[n]][57]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive5_7, send6_8, tempreceive5_7, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec5_7[nl[n]]), &(Buffersend6_8[nl[n]]), &(tempBufferrec5_7[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR6]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						-N3G, 0, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send5_7, receive5_7, tempreceive5_7, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend5_7[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]]), &(Bufferrec5_7[nl[n]]), &(tempBufferrec5_7[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR6]][AMR_PARENT]]][57]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR6]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
		//else fprintf(stderr, "Error in indexing!\n");
	}

	//Negative X3
	if (block[n][AMR_NBR5] >= 0){
		if (block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1){
			if (block[block[n][AMR_NBR5]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[block[n][AMR_NBR5]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL] && nstep%block[block[n][AMR_NBR5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][60], &Statbound[nl[n]][60]);
				}
				unpack_receive3(n, block[n][AMR_NBR5], 0, -N1G, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, BS_3, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), receive6, tempreceive6, prim,
					&(Bufferp[nl[n]]), &(Bufferrec6[nl[n]]), &(tempBufferrec6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[n][AMR_NBR5], 0, -N1G, BS_1 + N1G, 0, -N2G, BS_2 + N2G, 0, BS_3, BS_3 + N3G, (BS_1 + 2 * N1G), (BS_2 + 2 * N2G), send6, tempreceive6, prim,
					&(Bufferp[nl[n]]), &(Buffersend6[nl[block[n][AMR_NBR5]]]), &(tempBufferrec6[nl[n]]), &(boundevent1[nl[block[n][AMR_NBR5]]][60]), NULL, block[n][AMR_TIMELEVEL] < block[block[n][AMR_NBR5]][AMR_TIMELEVEL]);
			}
		}
		else if (block[block[n][AMR_NBR5]][AMR_REFINED] == 1){
			//receive from finer grid
			if (block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_NODE] != block[n][AMR_NODE]){
				if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
					MPI_Wait(&boundreqs[nl[n]][62], &Statbound[nl[n]][62]);
				}
				unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, BS_3, BS_3 + N3G,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive6_2, tempreceive6_2, prim,
					&(Bufferp[nl[n]]), &(Bufferrec6_2[nl[n]]), &(tempBufferrec6_2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			else{
				unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD1], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, BS_3, BS_3 + N3G,
					(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send6, tempreceive6_2, prim,
					&(Bufferp[nl[n]]), &(Buffersend6[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]]), &(tempBufferrec6_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_CHILD1]]][60]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL]);
			}
			if (REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][64], &Statbound[nl[n]][64]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive6_4, tempreceive6_4, prim,
						&(Bufferp[nl[n]]), &(Bufferrec6_4[nl[n]]), &(tempBufferrec6_4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD3], 0, -N1G / (1 + REF_1), BS_1 / (1 + REF_1) + (1 - REF_1)*N1G, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send6, tempreceive6_4, prim,
						&(Bufferp[nl[n]]), &(Buffersend6[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]]), &(tempBufferrec6_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_CHILD3]]][60]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD3]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][66], &Statbound[nl[n]][66]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive6_6, tempreceive6_6, prim,
						&(Bufferp[nl[n]]), &(Bufferrec6_6[nl[n]]), &(tempBufferrec6_6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD5], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 0, -N2G / (1 + REF_2), BS_2 / (1 + REF_2) + (1 - REF_2)*N2G, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send6, tempreceive6_6, prim,
						&(Bufferp[nl[n]]), &(Buffersend6[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]]), &(tempBufferrec6_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_CHILD5]]][60]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL]);
				}
			}
			if (REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][68], &Statbound[nl[n]][68]);
					}
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), receive6_8, tempreceive6_8, prim,
						&(Bufferp[nl[n]]), &(Bufferrec6_8[nl[n]]), &(tempBufferrec6_8[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive3(n, block[block[n][AMR_NBR5]][AMR_CHILD7], 1, BS_1 / (1 + REF_1), BS_1 + D1*REF_1, 1, BS_2 / (1 + REF_2), BS_2 + D2*REF_2, 0, BS_3, BS_3 + N3G,
						(BS_1 + 2 * N1G) / (1 + REF_1), (BS_2 + 2 * N2G) / (1 + REF_2), send6, tempreceive6_8, prim,
						&(Bufferp[nl[n]]), &(Buffersend6[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]]), &(tempBufferrec6_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_CHILD7]]][60]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_CHILD7]][AMR_TIMELEVEL]);
				}
			}
		}
		else if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1){
			//receive from coarser grid
			if (block[block[n][AMR_PARENT]][AMR_CHILD2] == n){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][62], &Statbound[nl[n]][62]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive6_2, send5_1, tempreceive6_2, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec6_2[nl[n]]), &(Buffersend5_1[nl[n]]), &(tempBufferrec6_2[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_2, receive6_2, tempreceive6_2, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_2[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]]), &(Bufferrec6_2[nl[n]]), &(tempBufferrec6_2[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]][62]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD4] == n && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][64], &Statbound[nl[n]][64]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive6_4, send5_3, tempreceive6_4, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec6_4[nl[n]]), &(Buffersend5_3[nl[n]]), &(tempBufferrec6_4[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_4, receive6_4, tempreceive6_4, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_4[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]]), &(Bufferrec6_4[nl[n]]), &(tempBufferrec6_4[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]][64]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD6] == n && REF_1 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][66], &Statbound[nl[n]][66]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive6_6, send5_5, tempreceive6_6, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec6_6[nl[n]]), &(Buffersend5_5[nl[n]]), &(tempBufferrec6_6[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_6, receive6_6, tempreceive6_6, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_6[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]]), &(Bufferrec6_6[nl[n]]), &(tempBufferrec6_6[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]][66]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
			if (block[block[n][AMR_PARENT]][AMR_CHILD8] == n && REF_1 == 1 && REF_2 == 1){
				if (block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_NODE] != block[n][AMR_NODE]){
					if (nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] - 1 || nstep == -1 || (PRESTEP2 && block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] && nstep%block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL] == block[n][AMR_TIMELEVEL] - 1)){
						MPI_Wait(&boundreqs[nl[n]][68], &Statbound[nl[n]][68]);
					}
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), receive6_8, send5_7, tempreceive6_8, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Bufferrec6_8[nl[n]]), &(Buffersend5_7[nl[n]]), &(tempBufferrec6_8[nl[n]]), NULL, NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
				else{
					unpack_receive_coarse3(n, block[block[n][AMR_NBR5]][AMR_PARENT], -N1G, BS_1 + N1G, -N2G, BS_2 + N2G,
						BS_3, BS_3 + N3G, (BS_1 / (1 + REF_1) + 2 * N1G), (BS_2 / (1 + REF_2) + 2 * N2G), send6_8, receive6_8, tempreceive6_8, prim, ps,
						&(Bufferp[nl[n]]), &(Bufferps[nl[n]]), &(Buffersend6_8[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]]), &(Bufferrec6_8[nl[n]]), &(tempBufferrec6_8[nl[n]]), &(boundevent1[nl[block[block[n][AMR_NBR5]][AMR_PARENT]]][68]), NULL, block[n][AMR_TIMELEVEL] < block[block[block[n][AMR_NBR5]][AMR_PARENT]][AMR_TIMELEVEL]);
				}
			}
		}
	}
#endif
}