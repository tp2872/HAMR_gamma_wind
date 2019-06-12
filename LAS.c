#include "decs_MPI.h"

/*Calculate for every block the timestep. This function should be node independent*/
void set_timelevel(int tag){
	int n;
	int i, j, z, l, ni, nj, nz;
	int task, n_block, min_timelevel;
	int min_j[NB_1*32];
	ni = NB_1;
	nj = NB_2;
	nz = NB_3;
	
	const int i_max = log(AMR_MAXTIMELEVEL) / log(2);
	if (nstep > 0){
		for (n = 0; n < n_active; n++){
			block[n_ord[n]][AMR_TIMELEVEL] = 1;
			for (i = i_max; i >= 0; i--){
				if (bdt[nl[n_ord[n]]][0] / dt >=1.05 * pow(2, i)){
					block[n_ord[n]][AMR_TIMELEVEL] = round(pow(2, i));
					break;
				}
			}
		}
	}

	//Send timelevel of all blocks to all nodes only when load balancing, otherwise send only to neighbouring nodes/blocks
	if (tag) {
		//First make sure all nodes have the same information regarding the timestep
		//Send for every block (l,i,j,z) to block (l2,i,j2,z2) on other nodes using non-blocking send
		for (n = 0; n < n_active_total; n++) {
			rc = MPI_Ibcast(&block[n_ord_total[n]][AMR_TIMELEVEL], 1, MPI_INT, block[n_ord_total[n]][AMR_NODE], mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
		}

		//Receive from other nodes using blocking receive
		for (n = 0; n < n_active_total; n++) {
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
		}

		//Fixate timelevel on block level and 0-level theta-phi slice
		/*for (n = 0; n < n_active_total; n++) {
			min_timelevel = 1000;
			if ((block[n_ord_total[n]][AMR_LEVEL] > 0) && (block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n])) {
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i++) {
					if (block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_TIMELEVEL] < min_timelevel) min_timelevel = block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_TIMELEVEL];
				}
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i++) {
					block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_TIMELEVEL] = min_timelevel;
				}
			}
			if ((block[n_ord_total[n]][AMR_LEVEL] == 0) && (block[n_ord_total[n]][AMR_COORD2]==0) && (block[n_ord_total[n]][AMR_COORD3] == 0)) {
				for (j = 0; j < NB_2; j++)for (z = 0; z < NB_3; z++) {
					n_block = AMR_coord_linear2(0, j, block[n_ord_total[n]][AMR_COORD1], j, z);
					if (block[n_block][AMR_TIMELEVEL] < min_timelevel)  min_timelevel = block[n_block][AMR_TIMELEVEL];
				}
				for (j = 0; j < NB_2; j++)for (z = 0; z < NB_3; z++) {
					n_block = AMR_coord_linear2(0, j, block[n_ord_total[n]][AMR_COORD1], j, z);
					block[n_block][AMR_TIMELEVEL]=min_timelevel;
				}
			}
		}*/

		//Fixate the timestep around the pole
		for (l = 0; l < N_LEVELS_3D; l++) {
			ni = NB_1 * pow(1 + REF_1, l);
			nj = NB_2 * pow(1 + REF_2, l);
			nz = NB_3 * pow(1 + REF_3 * (!DEREFINE_POLE), l);
			//#pragma omp parallel for schedule(static,1) private(i)
			for (i = 0; i < ni; i++) {
				if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_ACTIVE] == 1) {
					min_j[i] = 10000;
					if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 1 || block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 2 || block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 3) {
						for (z = 0; z < nz; z++) {
							min_j[i] = MY_MIN(block[AMR_coord_linear2(l, 0, i, 0, z)][AMR_TIMELEVEL], min_j[i]);
						}
						for (z = 0; z < nz; z++) {
							block[AMR_coord_linear2(l, 0, i, 0, z)][AMR_TIMELEVEL] = min_j[i];
						}
					}
				}
			}
			//#pragma omp parallel for schedule(static,1) private(i)
			for (i = 0; i < ni; i++) {
				if (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_ACTIVE] == 1) {
					min_j[i] = 10000;
					if (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_POLE] == 1 || block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_POLE] == 2 || block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_POLE] == 3) {
						for (z = 0; z < nz; z++) {
							min_j[i] = MY_MIN(block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z)][AMR_TIMELEVEL], min_j[i]);
						}
						for (z = 0; z < nz; z++) {
							block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z)][AMR_TIMELEVEL] = min_j[i];
						}
					}
				}
			}
		}
	}
	else {
		//First set the timestep around the poles
		for (n = 0; n < n_active; n++) {
			if (block[n_ord[n]][AMR_POLE] > 0) {
				for (z = 0; z < NB_3*pow(1 + REF_3 * (!DEREFINE_POLE), block[n_ord[n]][AMR_LEVEL3]); z++) {
					n_block = AMR_coord_linear2(block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_COORD2] / pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]), block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
					if (block[n_block][AMR_NODE] != rank) {
						MPI_Isend(&block[n_ord[n]][AMR_TIMELEVEL], 1, MPI_INT, block[n_block][AMR_NODE], (3 * NB_LOCAL + block[n_ord[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
						MPI_Irecv(&block[n_block][AMR_TIMELEVEL], 1, MPI_INT, block[n_block][AMR_NODE], (3 * NB_LOCAL + block[n_block][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_block]);
						MPI_Request_free(&req[0]);
					}
				}
			}
		}

		//Fixate the timestep around the pole
		for (n = 0; n < n_active; n++) {
			if (block[n_ord[n]][AMR_POLE] > 0) {
				min_timelevel = 1000;
				for (z = 0; z < NB_3*pow(1 + REF_3 * (!DEREFINE_POLE), block[n_ord[n]][AMR_LEVEL3]); z++) {
					n_block = AMR_coord_linear2(block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_COORD2] / pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]), block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
					if (block[n_block][AMR_NODE] != rank) {
						MPI_Wait(&request_timelevel[n_block], &Statbound[0][0]);
					}
					if (block[n_block][AMR_TIMELEVEL] < min_timelevel) min_timelevel = block[n_block][AMR_TIMELEVEL];
				}
				for (z = 0; z < NB_3*pow(1 + REF_3 * (!DEREFINE_POLE), block[n_ord[n]][AMR_LEVEL3]); z++) {
					n_block = AMR_coord_linear2(block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_COORD2] / pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]), block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
					block[n_block][AMR_TIMELEVEL] = min_timelevel;
				}
			}
		}

		//Now set timelevel for all neighbours and corners
		for (n = 0; n < n_active; n++) {
			for (i = AMR_NBR1; i <= AMR_CORN12; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					MPI_Isend(&block[n_ord[n]][AMR_TIMELEVEL], 1, MPI_INT, block[block[n_ord[n]][i]][AMR_NODE], (3 * NB_LOCAL + block[n_ord[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
					MPI_Irecv(&block[block[n_ord[n]][i]][AMR_TIMELEVEL], 1, MPI_INT, block[block[n_ord[n]][i]][AMR_NODE], (3 * NB_LOCAL + block[block[n_ord[n]][i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[block[n_ord[n]][i]]);
					MPI_Request_free(&req[0]);
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					block[block[n_ord[n]][i]][AMR_TAG1] = 0;
					block[block[n_ord[n]][i]][AMR_TAG3] = 0;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					if (block[block[n_ord[n]][i]][AMR_TAG1] == 0) {
						MPI_Isend(&block[n_ord[n]][AMR_TIMELEVEL], 1, MPI_INT, block[block[n_ord[n]][i]][AMR_NODE], (3 * NB_LOCAL + block[n_ord[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
						MPI_Request_free(&req[0]);
						block[block[n_ord[n]][i]][AMR_TAG1] = 1;
					}
					if (block[block[n_ord[n]][i]][AMR_TAG3] == 0) {
						MPI_Irecv(&block[block[n_ord[n]][i]][AMR_TIMELEVEL], 1, MPI_INT, block[block[n_ord[n]][i]][AMR_NODE], (3 * NB_LOCAL + block[block[n_ord[n]][i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[block[n_ord[n]][i]]);
						block[block[n_ord[n]][i]][AMR_TAG3] = 1;
					}
				}
			}
		}
		for (n = 0; n < n_active; n++) {
			for (i = AMR_NBR1; i <= AMR_CORN12; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					MPI_Wait(&request_timelevel[block[n_ord[n]][i]], &Statbound[0][0]);
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					block[block[n_ord[n]][i]][AMR_TAG3] = 0;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_NODE] != rank) {
					if (block[block[n_ord[n]][i]][AMR_TAG3] == 0) {
						MPI_Wait(&request_timelevel[block[n_ord[n]][i]], &Statbound[0][0]);
						block[block[n_ord[n]][i]][AMR_TAG3] = 1;
					}
				}
			}
		}
	}

	//Create communicators for nodes which have a minimum (i) timelevel
	set_communicator();
	set_corners(tag);
}

void set_prestep(void){
	int n;

	#if(PRESTEP || PRESTEP2)
	int timelevel_min[N_GPU];
	int blocks_per_timestep[N_GPU];
	int blocks_this_timestep[N_GPU];

	for (n = 0; n < N_GPU; n++) {
		blocks_this_timestep[n] = 0;
		timelevel_min[n] = AMR_MAXTIMELEVEL;
		blocks_per_timestep[n] = 0;
	}

	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	//Find the minimum timelevel on this node
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_TIMELEVEL] < timelevel_min[block[n_ord[n]][AMR_GPU]]) timelevel_min[block[n_ord[n]][AMR_GPU]] = block[n_ord[n]][AMR_TIMELEVEL];
	}

	//Calculate the number of blocks you want to evolve simultaneously
	for (n = 0; n < N_GPU; n++) blocks_per_timestep[n] = (count_gpu[n] - count_gpu[n] % (AMR_MAXTIMELEVEL / timelevel_min[n])) / (AMR_MAXTIMELEVEL / timelevel_min[n]);
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
	}

	//First make sure that the block required for prestepping in case of 2nd order time accuracy are preevolved
	#if(PRESTEP2)
	int i, j;
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && block[n_ord[n]][AMR_PRESTEP] == 0 && block[n_ord[n]][AMR_TIMELEVEL]>1){
			for (i = AMR_NBR1; i <= AMR_NBR6; i++){
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
					break;
				}
			}
			for (i = AMR_NBR1P; i <= AMR_NBR6P; i++){
				if (block[n_ord[n]][i] >= 0 && block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
					break;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_NBR6_8; i++){
				if (block[n_ord[n]][i] >= 0 && block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
					break;
				}
			}
			if (block[n_ord[n]][AMR_POLE] > 0){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
			}
			if (block[n_ord[n]][AMR_PRESTEP] == 1) block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
		}
	}
	#endif

	//If you don't have sufficient blocks this timestep preevolve some blocks if available
	for (n = 0; n < n_active; n++){
		if ((nstep %  timelevel_min[block[n_ord[n]][AMR_GPU]]) == timelevel_min[block[n_ord[n]][AMR_GPU]] - 1){
			if (blocks_this_timestep[block[n_ord[n]][AMR_GPU]] < blocks_per_timestep[block[n_ord[n]][AMR_GPU]] && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) != block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0 && (block[n_ord[n]][AMR_POLE] == 0)){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
				blocks_this_timestep[block[n_ord[n]][AMR_GPU]]++;
			}
		}
	}

	#if(!PRESTEP2)
	//If at end of switchtimelevel do not pre-evolve
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) >= 2 * AMR_SWITCHTIMELEVEL - 2 * AMR_MAXTIMELEVEL){
			block[n_ord[n]][AMR_PRESTEP] = 0;
			block[n_ord[n]][AMR_NSTEP] = nstep;
		}
	}
	#endif
	#else
	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	#endif
}

void prestep_bound(void){
	int flag, n;
	//If block is prestepped send non-corrected boundary cells to blocks with finer timelevels for interpolation in time
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_boundprim1(1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_boundprim2(1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) {
		#if(N_GPU>1)
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		if (prestep_full[nl[n_ord[n]]] == 1) {
			bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
			bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
			#if(N3>1)
			bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
			#endif
		}
		else if (prestep_half[nl[n_ord[n]]] == 1) {
			bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
			bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
			#if(N3>1)
			bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
			#endif
		}
	}

	set_iprobe(0, &flag);
	do{
		//Store difference between evolved and required flux/electric field in temporary array
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif	
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			#if(N3G>0)
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif		
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif	
			E_rec1(E_corn, BufferE_1, n_ord[n], 1);
			E_rec2(E_corn, BufferE_1, n_ord[n], 1);
			#if(N3G>0)
			E_rec3(E_corn, BufferE_1, n_ord[n], 1);
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		#if(N_GPU>1)
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		#if(N3G>0)	
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
		#endif
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}

	//Then reset flux and electric fields to zero
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) {
		if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) GPU_cleanup_post(n_ord[n]);
	}

	//Then insert flux differnce from temporary array in zeroed out flux and electric fields arrays
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)){
		#if(N_GPU>1)
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		flux_rec1(F1, BufferF1_1, n_ord[n], 6);
		flux_rec2(F2, BufferF2_1, n_ord[n], 6);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 6);
		#endif

		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
		E_rec1(E_corn, BufferE_1, n_ord[n], 6);
		E_rec2(E_corn, BufferE_1, n_ord[n], 6);
		#if(N3G>0)
		E_rec3(E_corn, BufferE_1, n_ord[n], 6);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
		#endif
	}

	//Evolve magnetic fields at boundary
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif	
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) {
		GPU_consttransport3_post(dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}

	//Evolve conserved quantities at boundary using update fluxes and invert to primitive variables plus floor
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) {
		GPU_fixup_post(n_ord[n], dt*(double)block[n_ord[n]][AMR_TIMELEVEL]);
	}
}

void set_corners(int tag){
	int n,i;
	int counter0, counter1, counter2, counter3;
	int counter0_1, counter1_1, counter2_1, counter3_1;
	int counter0_2, counter1_2, counter2_2, counter3_2;
	int temp;

	if (tag == 0) {
		#pragma omp parallel for schedule(static, 1) private(n,i)
		for (n = 0; n < n_active_total; n++) {
			block[n_ord_total[n]][AMR_TAG] = 0;
		}

		#pragma omp parallel for schedule(static, 1) private(n,i)
		for (n = 0; n < n_active; n++) {
			block[n_ord[n]][AMR_TAG] = 1;
			for (i = AMR_NBR1; i <= AMR_CORN12; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1) {
					block[block[n_ord[n]][i]][AMR_TAG] = 1;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1) {
					block[block[n_ord[n]][i]][AMR_TAG] = 1;
				}
			}
		}
	}
	else {
		#pragma omp parallel for schedule(static, 1)
		for (n = 0; n < n_active_total; n++) {
			block[n_ord_total[n]][AMR_TAG] = 1;
		}
	}

	//Set the most important corner value of the electric field to break the degeneracy of E-fields at each corner
	#if(TIMESTEP_JET)
	for (n = 0; n < n_active_total; n++){
		block[n_ord_total[n]][AMR_CORN1D] = -100;
		block[n_ord_total[n]][AMR_CORN1D_1] = -100;
		block[n_ord_total[n]][AMR_CORN1D_2] = -100;
		block[n_ord_total[n]][AMR_CORN2D] = -100;
		block[n_ord_total[n]][AMR_CORN2D_1] = -100;
		block[n_ord_total[n]][AMR_CORN2D_2] = -100;
		block[n_ord_total[n]][AMR_CORN3D] = -100;
		block[n_ord_total[n]][AMR_CORN3D_1] = -100;
		block[n_ord_total[n]][AMR_CORN3D_2] = -100;
		block[n_ord_total[n]][AMR_CORN4D] = -100;
		block[n_ord_total[n]][AMR_CORN4D_1] = -100;
		block[n_ord_total[n]][AMR_CORN4D_2] = -100;
		block[n_ord_total[n]][AMR_CORN5D] = -100;
		block[n_ord_total[n]][AMR_CORN5D_1] = -100;
		block[n_ord_total[n]][AMR_CORN5D_2] = -100;
		block[n_ord_total[n]][AMR_CORN6D] = -100;
		block[n_ord_total[n]][AMR_CORN6D_1] = -100;
		block[n_ord_total[n]][AMR_CORN6D_2] = -100;
		block[n_ord_total[n]][AMR_CORN7D] = -100;
		block[n_ord_total[n]][AMR_CORN7D_1] = -100;
		block[n_ord_total[n]][AMR_CORN7D_2] = -100;
		block[n_ord_total[n]][AMR_CORN8D] = -100;
		block[n_ord_total[n]][AMR_CORN8D_1] = -100;
		block[n_ord_total[n]][AMR_CORN8D_2] = -100;
		block[n_ord_total[n]][AMR_CORN9D] = -100;
		block[n_ord_total[n]][AMR_CORN9D_1] = -100;
		block[n_ord_total[n]][AMR_CORN9D_2] = -100;
		block[n_ord_total[n]][AMR_CORN10D] = -100;
		block[n_ord_total[n]][AMR_CORN10D_1] = -100;
		block[n_ord_total[n]][AMR_CORN10D_2] = -100;
		block[n_ord_total[n]][AMR_CORN11D] = -100;
		block[n_ord_total[n]][AMR_CORN11D_1] = -100;
		block[n_ord_total[n]][AMR_CORN11D_2] = -100;
		block[n_ord_total[n]][AMR_CORN12D] = -100;
		block[n_ord_total[n]][AMR_CORN12D_1] = -100;
		block[n_ord_total[n]][AMR_CORN12D_2] = -100;
	}
	#else
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			//Corn 1
			block[n_ord_total[n]][AMR_CORN1D] = -10;
			block[n_ord_total[n]][AMR_CORN1D_1] = -10;
			block[n_ord_total[n]][AMR_CORN1D_2] = -10;

			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				block[n_ord_total[n]][AMR_CORN1D] = n_ord_total[n];

				//Corn 1
				counter0 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter0_1 = AMR_MAXTIMELEVEL;
				counter0_2 = AMR_MAXTIMELEVEL;
				counter1 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter1_1 = AMR_MAXTIMELEVEL;
				counter1_2 = AMR_MAXTIMELEVEL;
				counter2 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter2_1 = AMR_MAXTIMELEVEL;
				counter2_2 = AMR_MAXTIMELEVEL;
				counter3_1 = AMR_MAXTIMELEVEL;
				counter3_2 = AMR_MAXTIMELEVEL;

				if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3) {
					if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1) {
						counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL];
						if (counter1 > counter0) {
							block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_NBR1];
						}
					}
					else if (block[n_ord_total[n]][AMR_NBR1_7] >= 0 && block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_ACTIVE] == 1) {
						counter1_1 += 10000;
						counter1_2 += 10000;
						counter1 = 100000;
						block[n_ord_total[n]][AMR_CORN1D] = -2;
						counter1_1 -= (block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_LEVEL3]));
						counter1_2 -= (block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL3]));

						if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[n_ord_total[n]][AMR_NBR1_7];
						if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[n_ord_total[n]][AMR_NBR1_8];
					}
				}

				if (block[n_ord_total[n]][AMR_NBR2] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1) {
						counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL];
						if ((counter2 > counter1) && (counter2 > counter0))block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_NBR2];
					}
					else if (block[n_ord_total[n]][AMR_NBR2_1] >= 0 && block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
						counter2_1 += 10000;
						counter2_2 += 10000;
						counter2 = 100000;
						block[n_ord_total[n]][AMR_CORN1D] = -2;
						counter2_1 -= (block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL3]));
						counter2_2 -= (block[block[n_ord_total[n]][AMR_NBR2_2]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR2_2]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR2_2]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR2_2]][AMR_LEVEL3]));
						if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[n_ord_total[n]][AMR_NBR2_1];
						if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[n_ord_total[n]][AMR_NBR2_2];
					}
				}

				if (block[n_ord_total[n]][AMR_CORN1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3) {
					if (block[block[n_ord_total[n]][AMR_CORN1]][AMR_ACTIVE] == 1) {
						counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN1]][AMR_TIMELEVEL];
						if ((counter3 > counter2) && (counter3 > counter1) && (counter3 > counter0)) block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_CORN1];
					}
					else if (block[n_ord_total[n]][AMR_CORN1_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
						counter3_1 += 10000;
						counter3_2 += 10000;
						counter3 = 100000;
						block[n_ord_total[n]][AMR_CORN1D] = -2;
						counter3_1 -= (block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_LEVEL3]));
						counter3_2 -= (block[block[n_ord_total[n]][AMR_CORN1_2]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN1_2]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN1_2]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN1_2]][AMR_LEVEL3]));

						if ((counter3_1 > counter2_1) && (counter3_1 > counter1_1)) block[n_ord_total[n]][AMR_CORN1D_1] = block[n_ord_total[n]][AMR_CORN1_1];
						if ((counter3_2 > counter2_2) && (counter3_2 > counter1_2)) block[n_ord_total[n]][AMR_CORN1D_2] = block[n_ord_total[n]][AMR_CORN1_2];
					}
				}
			}
		}
	}
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			block[n_ord_total[n]][AMR_CORN2D] = -10;
			block[n_ord_total[n]][AMR_CORN2D_1] = -10;
			block[n_ord_total[n]][AMR_CORN2D_2] = -10;
			block[n_ord_total[n]][AMR_CORN3D] = -10;
			block[n_ord_total[n]][AMR_CORN3D_1] = -10;
			block[n_ord_total[n]][AMR_CORN3D_2] = -10;
			block[n_ord_total[n]][AMR_CORN4D] = -10;
			block[n_ord_total[n]][AMR_CORN4D_1] = -10;
			block[n_ord_total[n]][AMR_CORN4D_2] = -10;

			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				//Corn 2
				if (block[n_ord_total[n]][AMR_CORN2] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D];
						block[n_ord_total[n]][AMR_CORN2D_1] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D_1];
						block[n_ord_total[n]][AMR_CORN2D_2] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR3_5] >= 0 && block[block[n_ord_total[n]][AMR_NBR3_5]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN2D] = -2;
						block[n_ord_total[n]][AMR_CORN2D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR3_5]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_NBR3_5]][AMR_CORN1D_1]);
						block[n_ord_total[n]][AMR_CORN2D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN1D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN2P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]) {
							//block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN2D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN2D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN2D_1] = temp;
									block[n_ord_total[n]][AMR_CORN2D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN2D_1] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN2D_2] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN2D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN2D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN2D_1] = temp;
									block[n_ord_total[n]][AMR_CORN2D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN2D_1] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN2D_2] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN2D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN2D] = block[n_ord_total[n]][AMR_NBR2];
						if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D]; //good
					}
				}

				//Corn 3
				if (block[n_ord_total[n]][AMR_CORN3] >= 0) {
					if (block[block[n_ord_total[n]][AMR_CORN3]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D];
						block[n_ord_total[n]][AMR_CORN3D_1] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D_1];
						block[n_ord_total[n]][AMR_CORN3D_2] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D_2];
					}
					else if (block[n_ord_total[n]][AMR_CORN3_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN3D] = -2;
						block[n_ord_total[n]][AMR_CORN3D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN3_1]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_CORN3_1]][AMR_CORN1D_1]);
						block[n_ord_total[n]][AMR_CORN3D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN3_2]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_CORN3_2]][AMR_CORN1D_2]);
					}
					else if (block[n_ord_total[n]][AMR_CORN3P] >= 0 && block[block[n_ord_total[n]][AMR_CORN3P]][AMR_ACTIVE] == 1) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]) {
							//block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_1];
							temp = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN3D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN3D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN3D_1] = temp;
									block[n_ord_total[n]][AMR_CORN3D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN3D_1] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN3D_2] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_2];
							temp = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN3D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN3D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN3D_1] = temp;
									block[n_ord_total[n]][AMR_CORN3D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN3D_1] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN3D_2] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN3D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR3]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN3D] = block[n_ord_total[n]][AMR_NBR3];//good
						if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR4]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN3D] = block[n_ord_total[n]][AMR_NBR4];//good
						if (block[n_ord_total[n]][AMR_NBR3_1] >= 0 && block[block[n_ord_total[n]][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
							block[n_ord_total[n]][AMR_CORN3D] = -2;
							block[n_ord_total[n]][AMR_CORN3D_1] = block[n_ord_total[n]][AMR_NBR3_1];
							block[n_ord_total[n]][AMR_CORN3D_2] = block[n_ord_total[n]][AMR_NBR3_2];
						}
					}
				}

				//Corn 4
				if (block[n_ord_total[n]][AMR_CORN4] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D];
						block[n_ord_total[n]][AMR_CORN4D_1] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D_1];
						block[n_ord_total[n]][AMR_CORN4D_2] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR4_5] >= 0 && block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN4D] = -2;
						block[n_ord_total[n]][AMR_CORN4D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN1D_1]);
						block[n_ord_total[n]][AMR_CORN4D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR4_6]][AMR_CORN1D], block[block[n_ord_total[n]][AMR_NBR4_6]][AMR_CORN1D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN4P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]) {
							//block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN4D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN4D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN4D_1] = temp;
									block[n_ord_total[n]][AMR_CORN4D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN4D_1] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN4D_2] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN4D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN4D] = -2;
								if (block[temp][AMR_LEVEL3] - block[n_ord_total[n]][AMR_LEVEL3] == 0) {
									block[n_ord_total[n]][AMR_CORN4D_1] = temp;
									block[n_ord_total[n]][AMR_CORN4D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN4D_1] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_1];
									block[n_ord_total[n]][AMR_CORN4D_2] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN4D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D]; //good
						if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN4D] = block[n_ord_total[n]][AMR_NBR1]; //good
						if (block[n_ord_total[n]][AMR_NBR1_3] >= 0 && block[block[n_ord_total[n]][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
							block[n_ord_total[n]][AMR_CORN4D] = -2;
							block[n_ord_total[n]][AMR_CORN4D_1] = block[n_ord_total[n]][AMR_NBR1_3];
							block[n_ord_total[n]][AMR_CORN4D_2] = block[n_ord_total[n]][AMR_NBR1_4];
						}
					}
				}
				if (block[n_ord_total[n]][AMR_NBR4] < 0) {
					block[n_ord_total[n]][AMR_CORN4D] = -100;
					block[n_ord_total[n]][AMR_CORN4D_1] = -100;
					block[n_ord_total[n]][AMR_CORN4D_2] = -100;
					block[n_ord_total[n]][AMR_CORN3D] = -100;
					block[n_ord_total[n]][AMR_CORN3D_1] = -100;
					block[n_ord_total[n]][AMR_CORN3D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR2] < 0) {
					block[n_ord_total[n]][AMR_CORN1D] = -100;
					block[n_ord_total[n]][AMR_CORN1D_1] = -100;
					block[n_ord_total[n]][AMR_CORN1D_2] = -100;
					block[n_ord_total[n]][AMR_CORN2D] = -100;
					block[n_ord_total[n]][AMR_CORN2D_1] = -100;
					block[n_ord_total[n]][AMR_CORN2D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR1] < 0 || block[n_ord_total[n]][AMR_POLE] == 1 || block[n_ord_total[n]][AMR_POLE] == 3) {
					block[n_ord_total[n]][AMR_CORN1D] = -100;
					block[n_ord_total[n]][AMR_CORN1D_1] = -100;
					block[n_ord_total[n]][AMR_CORN1D_2] = -100;
					block[n_ord_total[n]][AMR_CORN4D] = -100;
					block[n_ord_total[n]][AMR_CORN4D_1] = -100;
					block[n_ord_total[n]][AMR_CORN4D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR3] < 0 || block[n_ord_total[n]][AMR_POLE] == 2 || block[n_ord_total[n]][AMR_POLE] == 3) {
					block[n_ord_total[n]][AMR_CORN3D] = -100;
					block[n_ord_total[n]][AMR_CORN3D_1] = -100;
					block[n_ord_total[n]][AMR_CORN3D_2] = -100;
					block[n_ord_total[n]][AMR_CORN2D] = -100;
					block[n_ord_total[n]][AMR_CORN2D_1] = -100;
					block[n_ord_total[n]][AMR_CORN2D_2] = -100;
				}
			}
		}
	}
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			//Corn 5
			block[n_ord_total[n]][AMR_CORN5D] = -10;
			block[n_ord_total[n]][AMR_CORN5D_1] = -10;
			block[n_ord_total[n]][AMR_CORN5D_2] = -10;
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				block[n_ord_total[n]][AMR_CORN5D] = n_ord_total[n];

				//Corn 5
				counter0 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter0_1 = AMR_MAXTIMELEVEL;
				counter0_2 = AMR_MAXTIMELEVEL;
				counter1 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter1_1 = AMR_MAXTIMELEVEL;
				counter1_2 = AMR_MAXTIMELEVEL;
				counter2 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter2_1 = AMR_MAXTIMELEVEL;
				counter2_2 = AMR_MAXTIMELEVEL;
				counter3_1 = AMR_MAXTIMELEVEL;
				counter3_2 = AMR_MAXTIMELEVEL;

				if (block[n_ord_total[n]][AMR_NBR6] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1) {
						counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL];
						if (counter1 >= counter0) {
							block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_NBR6];
						}
					}
					else if (block[n_ord_total[n]][AMR_NBR6_6] >= 0 && block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_ACTIVE] == 1) {
						counter1_1 += 10000;
						counter1_2 += 10000;
						counter1 = 100000;
						block[n_ord_total[n]][AMR_CORN5D] = -2;
						counter1_1 -= (block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_LEVEL3]));
						counter1_2 -= (block[block[n_ord_total[n]][AMR_NBR6_8]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR6_8]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR6_8]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR6_8]][AMR_LEVEL3]));

						if (counter1_1 >= counter0_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[n_ord_total[n]][AMR_NBR6_6];
						if (counter1_2 >= counter0_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[n_ord_total[n]][AMR_NBR6_8];
					}
				}

				if (block[n_ord_total[n]][AMR_NBR2] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1) {
						counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL];
						if ((counter2 > counter1) && (counter2 > counter0))block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_NBR2];
					}
					else if (block[n_ord_total[n]][AMR_NBR2_1] >= 0 && block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
						counter2_1 += 10000;
						counter2_2 += 10000;
						counter2 = 100000;
						block[n_ord_total[n]][AMR_CORN5D] = -2;
						counter2_1 -= (block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_LEVEL3]));
						counter2_2 -= (block[block[n_ord_total[n]][AMR_NBR2_3]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR2_3]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR2_3]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR2_3]][AMR_LEVEL3]));
						if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[n_ord_total[n]][AMR_NBR2_1];
						if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[n_ord_total[n]][AMR_NBR2_3];
					}
				}

				if (block[n_ord_total[n]][AMR_CORN5] >= 0) {
					if (block[block[n_ord_total[n]][AMR_CORN5]][AMR_ACTIVE] == 1) {
						counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN5]][AMR_TIMELEVEL];
						if ((counter3 >= counter2) && (counter3 > counter1) && (counter3 > counter0)) block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_CORN5];
					}
					else if (block[n_ord_total[n]][AMR_CORN5_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
						counter3_1 += 10000;
						counter3_2 += 10000;
						counter3 = 100000;
						block[n_ord_total[n]][AMR_CORN5D] = -2;
						counter3_1 -= (block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_LEVEL3]));
						counter3_2 -= (block[block[n_ord_total[n]][AMR_CORN5_2]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN5_2]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN5_2]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN5_2]][AMR_LEVEL3]));

						if ((counter3_1 >= counter2_1) && (counter3_1 > counter1_1)) block[n_ord_total[n]][AMR_CORN5D_1] = block[n_ord_total[n]][AMR_CORN5_1];
						if ((counter3_2 >= counter2_2) && (counter3_2 > counter1_2)) block[n_ord_total[n]][AMR_CORN5D_2] = block[n_ord_total[n]][AMR_CORN5_2];
					}
				}
			}
		}
	}

	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			block[n_ord_total[n]][AMR_CORN6D] = -10;
			block[n_ord_total[n]][AMR_CORN6D_1] = -10;
			block[n_ord_total[n]][AMR_CORN6D_2] = -10;
			block[n_ord_total[n]][AMR_CORN7D] = -10;
			block[n_ord_total[n]][AMR_CORN7D_1] = -10;
			block[n_ord_total[n]][AMR_CORN7D_2] = -10;
			block[n_ord_total[n]][AMR_CORN8D] = -10;
			block[n_ord_total[n]][AMR_CORN8D_1] = -10;
			block[n_ord_total[n]][AMR_CORN8D_2] = -10;
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				//Corn 6
				if (block[n_ord_total[n]][AMR_CORN6] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D];
						block[n_ord_total[n]][AMR_CORN6D_1] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D_1];
						block[n_ord_total[n]][AMR_CORN6D_2] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR5_5] >= 0 && block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN6D] = -2;
						block[n_ord_total[n]][AMR_CORN6D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_CORN5D_1]);
						block[n_ord_total[n]][AMR_CORN6D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR5_7]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_NBR5_7]][AMR_CORN5D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR5P] >= 0 && block[block[n_ord_total[n]][AMR_NBR5P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN6P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]) {
							//block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN6D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN6D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN6D_1] = temp;
									block[n_ord_total[n]][AMR_CORN6D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN6D_1] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN6D_2] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN6D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN6D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN6D_1] = temp;
									block[n_ord_total[n]][AMR_CORN6D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN6D_1] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN6D_2] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN6D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN6D] = block[n_ord_total[n]][AMR_NBR2];
						if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D]; //good
					}
				}

				//Corn 7
				if (block[n_ord_total[n]][AMR_CORN7] >= 0) {
					if (block[block[n_ord_total[n]][AMR_CORN7]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D];
						block[n_ord_total[n]][AMR_CORN7D_1] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D_1];
						block[n_ord_total[n]][AMR_CORN7D_2] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D_2];
					}
					else if (block[n_ord_total[n]][AMR_CORN7_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN7D] = -2;
						block[n_ord_total[n]][AMR_CORN7D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN7_1]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_CORN7_1]][AMR_CORN5D_1]);
						block[n_ord_total[n]][AMR_CORN7D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN7_2]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_CORN7_2]][AMR_CORN5D_2]);
					}
					else if (block[n_ord_total[n]][AMR_CORN7P] >= 0 && block[block[n_ord_total[n]][AMR_CORN7P]][AMR_ACTIVE] == 1) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]) {
							//block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_1];
							temp = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN7D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN7D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN7D_1] = temp;
									block[n_ord_total[n]][AMR_CORN7D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN7D_1] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN7D_2] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_2];
							temp = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN7D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN7D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN7D_1] = temp;
									block[n_ord_total[n]][AMR_CORN7D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN7D_1] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN7D_2] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN7D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN7D] = block[n_ord_total[n]][AMR_NBR5];//good
						if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR4]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN7D] = block[n_ord_total[n]][AMR_NBR4];//good
					}
				}

				//Corn 8
				if (block[n_ord_total[n]][AMR_CORN8] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D];
						block[n_ord_total[n]][AMR_CORN8D_1] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D_1];
						block[n_ord_total[n]][AMR_CORN8D_2] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR4_5] >= 0 && block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN8D] = -2;
						block[n_ord_total[n]][AMR_CORN8D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN5D_1]);
						block[n_ord_total[n]][AMR_CORN8D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR4_7]][AMR_CORN5D], block[block[n_ord_total[n]][AMR_NBR4_7]][AMR_CORN5D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN8P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]) {
							//block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN8D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN8D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN8D_1] = temp;
									block[n_ord_total[n]][AMR_CORN8D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN8D_1] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN8D_2] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN8D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN8D] = -2;
								if (block[temp][AMR_LEVEL2] == block[n_ord_total[n]][AMR_LEVEL2]) {
									block[n_ord_total[n]][AMR_CORN8D_1] = temp;
									block[n_ord_total[n]][AMR_CORN8D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN8D_1] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_1];
									block[n_ord_total[n]][AMR_CORN8D_2] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN8D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D]; //good
						if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN8D] = block[n_ord_total[n]][AMR_NBR6]; //good
					}
				}

				if (block[n_ord_total[n]][AMR_NBR4] < 0) {
					block[n_ord_total[n]][AMR_CORN8D] = -100;
					block[n_ord_total[n]][AMR_CORN8D_1] = -100;
					block[n_ord_total[n]][AMR_CORN8D_2] = -100;
					block[n_ord_total[n]][AMR_CORN7D] = -100;
					block[n_ord_total[n]][AMR_CORN7D_1] = -100;
					block[n_ord_total[n]][AMR_CORN7D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR2] < 0) {
					block[n_ord_total[n]][AMR_CORN5D] = -100;
					block[n_ord_total[n]][AMR_CORN5D_1] = -100;
					block[n_ord_total[n]][AMR_CORN5D_2] = -100;
					block[n_ord_total[n]][AMR_CORN6D] = -100;
					block[n_ord_total[n]][AMR_CORN6D_1] = -100;
					block[n_ord_total[n]][AMR_CORN6D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR6] < 0) {
					block[n_ord_total[n]][AMR_CORN5D] = -100;
					block[n_ord_total[n]][AMR_CORN5D_1] = -100;
					block[n_ord_total[n]][AMR_CORN5D_2] = -100;
					block[n_ord_total[n]][AMR_CORN8D] = -100;
					block[n_ord_total[n]][AMR_CORN8D_1] = -100;
					block[n_ord_total[n]][AMR_CORN8D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR5] < 0) {
					block[n_ord_total[n]][AMR_CORN7D] = -100;
					block[n_ord_total[n]][AMR_CORN7D_1] = -100;
					block[n_ord_total[n]][AMR_CORN7D_2] = -100;
					block[n_ord_total[n]][AMR_CORN6D] = -100;
					block[n_ord_total[n]][AMR_CORN6D_1] = -100;
					block[n_ord_total[n]][AMR_CORN6D_2] = -100;
				}
			}
		}
	}
	//Set the most important corner value of the electric field to break the degeneracy of E-fields at each corner
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			//Corn 9
			block[n_ord_total[n]][AMR_CORN9D] = -10;
			block[n_ord_total[n]][AMR_CORN9D_1] = -10;
			block[n_ord_total[n]][AMR_CORN9D_2] = -10;
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				block[n_ord_total[n]][AMR_CORN9D] = n_ord_total[n];

				//Corn 1
				counter0 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter0_1 = AMR_MAXTIMELEVEL;
				counter0_2 = AMR_MAXTIMELEVEL;
				counter1 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter1_1 = AMR_MAXTIMELEVEL;
				counter1_2 = AMR_MAXTIMELEVEL;
				counter2 = AMR_MAXTIMELEVEL - block[n_ord_total[n]][AMR_TIMELEVEL];
				counter2_1 = AMR_MAXTIMELEVEL;
				counter2_2 = AMR_MAXTIMELEVEL;
				counter3_1 = AMR_MAXTIMELEVEL;
				counter3_2 = AMR_MAXTIMELEVEL;

				if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3) {
					if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1) {
						counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL];
						if (counter1 > counter0) {
							block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_NBR1];
						}
					}
					else if (block[n_ord_total[n]][AMR_NBR1_4] >= 0 && block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_ACTIVE] == 1) {
						counter1_1 += 10000;
						counter1_2 += 10000;
						counter1 = 100000;
						block[n_ord_total[n]][AMR_CORN9D] = -2;
						counter1_1 -= (block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_LEVEL3]));
						counter1_2 -= (block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_LEVEL3]));

						if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[n_ord_total[n]][AMR_NBR1_4];
						if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[n_ord_total[n]][AMR_NBR1_8];
					}
				}

				if (block[n_ord_total[n]][AMR_NBR5] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1) {
						counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL];
						if ((counter2 > counter1) && (counter2 > counter0))block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_NBR5];
					}
					else if (block[n_ord_total[n]][AMR_NBR5_1] >= 0 && block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
						counter2_1 += 10000;
						counter2_2 += 10000;
						counter2 = 100000;
						block[n_ord_total[n]][AMR_CORN9D] = -2;
						counter2_1 -= (block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_LEVEL3]));
						counter2_2 -= (block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_LEVEL3]));
						if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[n_ord_total[n]][AMR_NBR5_1];
						if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[n_ord_total[n]][AMR_NBR5_5];
					}
				}

				if (block[n_ord_total[n]][AMR_CORN9] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3) {
					if (block[block[n_ord_total[n]][AMR_CORN9]][AMR_ACTIVE] == 1) {
						counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN9]][AMR_TIMELEVEL];
						if ((counter3 > counter2) && (counter3 > counter1) && (counter3 > counter0)) block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_CORN9];
					}
					else if (block[n_ord_total[n]][AMR_CORN9_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
						counter3_1 += 10000;
						counter3_2 += 10000;
						counter3 = 100000;
						block[n_ord_total[n]][AMR_CORN9D] = -2;
						counter3_1 -= (block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_LEVEL3]));
						counter3_2 -= (block[block[n_ord_total[n]][AMR_CORN9_2]][AMR_TIMELEVEL] - 2 * AMR_MAXTIMELEVEL * (block[block[n_ord_total[n]][AMR_CORN9_2]][AMR_LEVEL1] + block[block[n_ord_total[n]][AMR_CORN9_2]][AMR_LEVEL2] + block[block[n_ord_total[n]][AMR_CORN9_2]][AMR_LEVEL3]));

						if ((counter3_1 > counter2_1) && (counter3_1 > counter1_1)) block[n_ord_total[n]][AMR_CORN9D_1] = block[n_ord_total[n]][AMR_CORN9_1];
						if ((counter3_2 > counter2_2) && (counter3_2 > counter1_2)) block[n_ord_total[n]][AMR_CORN9D_2] = block[n_ord_total[n]][AMR_CORN9_2];
					}
				}
			}
		}
	}

	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2, temp)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_TAG] == 1) {
			block[n_ord_total[n]][AMR_CORN10D] = -10;
			block[n_ord_total[n]][AMR_CORN10D_1] = -10;
			block[n_ord_total[n]][AMR_CORN10D_2] = -10;
			block[n_ord_total[n]][AMR_CORN11D] = -10;
			block[n_ord_total[n]][AMR_CORN11D_1] = -10;
			block[n_ord_total[n]][AMR_CORN11D_2] = -10;
			block[n_ord_total[n]][AMR_CORN12D] = -10;
			block[n_ord_total[n]][AMR_CORN12D_1] = -10;
			block[n_ord_total[n]][AMR_CORN12D_2] = -10;
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1) {
				//Corn 10
				if (block[n_ord_total[n]][AMR_CORN10] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D];
						block[n_ord_total[n]][AMR_CORN10D_1] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D_1];
						block[n_ord_total[n]][AMR_CORN10D_2] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR3_2] >= 0 && block[block[n_ord_total[n]][AMR_NBR3_2]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN10D] = -2;
						block[n_ord_total[n]][AMR_CORN10D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR3_2]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_NBR3_2]][AMR_CORN9D_1]);
						block[n_ord_total[n]][AMR_CORN10D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN9D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN10P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]) {
							//block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN10D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN10D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN10D_1] = temp;
									block[n_ord_total[n]][AMR_CORN10D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN10D_1] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN10D_2] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN10D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN10D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN10D_1] = temp;
									block[n_ord_total[n]][AMR_CORN10D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN10D_1] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN10D_2] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN10D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN10D] = block[n_ord_total[n]][AMR_NBR5];
						if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D]; //good
					}
				}

				//Corn 11
				if (block[n_ord_total[n]][AMR_CORN11] >= 0) {
					if (block[block[n_ord_total[n]][AMR_CORN11]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D];
						block[n_ord_total[n]][AMR_CORN11D_1] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D_1];
						block[n_ord_total[n]][AMR_CORN11D_2] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D_2];
					}
					else if (block[n_ord_total[n]][AMR_CORN11_1] >= 0 && block[block[n_ord_total[n]][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN11D] = -2;
						block[n_ord_total[n]][AMR_CORN11D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN11_1]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_CORN11_1]][AMR_CORN9D_1]);
						block[n_ord_total[n]][AMR_CORN11D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_CORN11_2]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_CORN11_2]][AMR_CORN9D_2]);
					}
					else if (block[n_ord_total[n]][AMR_CORN11P] >= 0 && block[block[n_ord_total[n]][AMR_CORN11P]][AMR_ACTIVE] == 1) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]) {
							//block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_1];
							temp = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN11D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN11D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN11D_1] = temp;
									block[n_ord_total[n]][AMR_CORN11D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN11D_1] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN11D_2] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_2];
							temp = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN11D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN11D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN11D_1] = temp;
									block[n_ord_total[n]][AMR_CORN11D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN11D_1] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN11D_2] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN11D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR3]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN11D] = block[n_ord_total[n]][AMR_NBR3];//good
						if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN11D] = block[n_ord_total[n]][AMR_NBR6];//good
					}
				}

				//Corn 12
				if (block[n_ord_total[n]][AMR_CORN12] >= 0) {
					if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D];
						block[n_ord_total[n]][AMR_CORN12D_1] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D_1];
						block[n_ord_total[n]][AMR_CORN12D_2] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D_2];
					}
					else if (block[n_ord_total[n]][AMR_NBR6_2] >= 0 && block[block[n_ord_total[n]][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
						block[n_ord_total[n]][AMR_CORN12D] = -2;
						block[n_ord_total[n]][AMR_CORN12D_1] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR6_2]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_NBR6_2]][AMR_CORN9D_1]);
						block[n_ord_total[n]][AMR_CORN12D_2] = MY_MAX(block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_CORN9D], block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_CORN9D_2]);
					}
					else if (block[n_ord_total[n]][AMR_NBR6P] >= 0 && block[block[n_ord_total[n]][AMR_NBR6P]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_CORN12P] >= 0) {
						if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]) {
							//block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_1];
							temp = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_1];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN12D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN12D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN12D_1] = temp;
									block[n_ord_total[n]][AMR_CORN12D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN12D_1] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN12D_2] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_2];
								}
							}
						}
						else {
							//block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_2];
							temp = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_2];
							if ((temp < 0) || (block[temp][AMR_LEVEL1] + block[temp][AMR_LEVEL3] == block[n_ord_total[n]][AMR_LEVEL1] + block[n_ord_total[n]][AMR_LEVEL3])) block[n_ord_total[n]][AMR_CORN12D] = temp;
							else {
								block[n_ord_total[n]][AMR_CORN12D] = -2;
								if (block[temp][AMR_LEVEL1] == block[n_ord_total[n]][AMR_LEVEL1]) {
									block[n_ord_total[n]][AMR_CORN12D_1] = temp;
									block[n_ord_total[n]][AMR_CORN12D_2] = temp;
								}
								else {
									block[n_ord_total[n]][AMR_CORN12D_1] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_1];
									block[n_ord_total[n]][AMR_CORN12D_2] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_2];
								}
							}
						}
					}
					else
					{
						block[n_ord_total[n]][AMR_CORN12D] = n_ord_total[n];
						if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D]; //good
						if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN12D] = block[n_ord_total[n]][AMR_NBR1]; //good
					}
				}
				if (block[n_ord_total[n]][AMR_NBR6] < 0) {
					block[n_ord_total[n]][AMR_CORN12D] = -100;
					block[n_ord_total[n]][AMR_CORN12D_1] = -100;
					block[n_ord_total[n]][AMR_CORN12D_2] = -100;
					block[n_ord_total[n]][AMR_CORN11D] = -100;
					block[n_ord_total[n]][AMR_CORN11D_1] = -100;
					block[n_ord_total[n]][AMR_CORN11D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR5] < 0) {
					block[n_ord_total[n]][AMR_CORN9D] = -100;
					block[n_ord_total[n]][AMR_CORN9D_1] = -100;
					block[n_ord_total[n]][AMR_CORN9D_2] = -100;
					block[n_ord_total[n]][AMR_CORN10D] = -100;
					block[n_ord_total[n]][AMR_CORN10D_1] = -100;
					block[n_ord_total[n]][AMR_CORN10D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR1] < 0 || block[n_ord_total[n]][AMR_POLE] == 1 || block[n_ord_total[n]][AMR_POLE] == 3) {
					block[n_ord_total[n]][AMR_CORN9D] = -100;
					block[n_ord_total[n]][AMR_CORN9D_1] = -100;
					block[n_ord_total[n]][AMR_CORN9D_2] = -100;
					block[n_ord_total[n]][AMR_CORN12D] = -100;
					block[n_ord_total[n]][AMR_CORN12D_1] = -100;
					block[n_ord_total[n]][AMR_CORN12D_2] = -100;
				}
				if (block[n_ord_total[n]][AMR_NBR3] < 0 || block[n_ord_total[n]][AMR_POLE] == 2 || block[n_ord_total[n]][AMR_POLE] == 3) {
					block[n_ord_total[n]][AMR_CORN11D] = -100;
					block[n_ord_total[n]][AMR_CORN11D_1] = -100;
					block[n_ord_total[n]][AMR_CORN11D_2] = -100;
					block[n_ord_total[n]][AMR_CORN10D] = -100;
					block[n_ord_total[n]][AMR_CORN10D_1] = -100;
					block[n_ord_total[n]][AMR_CORN10D_2] = -100;
				}
			}
		}
	}
#endif
	
	//int test = AMR_coord_linear2(1, 0, 3, 1, 0);
	//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	//test = block[test][AMR_NBR1_3];
	//fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);

	/*int test = AMR_coord_linear2(1, 0, 1, 1, 0);
	fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = block[test][AMR_CORN2D_2];
	fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = AMR_coord_linear2(1, 1, 0, 1, 0);
	fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = block[test][AMR_CORN1D_2];
	fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = AMR_coord_linear2(2, 1,2,2,1);
	fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = block[test][AMR_CORN4D];
	fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = AMR_coord_linear2(1, 0, 2, 1, 0);
	fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	test = block[test][AMR_CORN3D_2];
	fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	*/
	//if (test != block[AMR_coord_linear2(0, 5, 0, 5, 0)][AMR_CORN1D_1] || test != block[AMR_coord_linear2(1, 4, 0, 4, 0)][AMR_CORN2D_1] || test != block[AMR_coord_linear2(1, 4, 1, 4, 0)][AMR_CORN3D_1] || test != block[test][AMR_CORN4D])fprintf(stderr, "corner error \n");
	//fprintf(stderr, "corner %d %d %d %d\n", block[AMR_coord_linear2(0, 5, 0, 5, 0)][AMR_CORN1D], block[AMR_coord_linear2(1, 4, 0, 4, 0)][AMR_CORN2D], block[AMR_coord_linear2(1, 4, 1, 4, 0)][AMR_CORN3D], block[test][AMR_CORN4D_1]);
	//n = AMR_coord_linear2(1, 5, 2, 10, 1);
	//test = block[n][AMR_CORN3D];
	//fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	//test = block[n][AMR_CORN3D_2];
	//fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	//n = AMR_coord_linear2(1, 5, 2, 11, 1);
	//test = block[n][AMR_CORN4D];
	//fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	//test = block[n][AMR_CORN4D_2];
	//fprintf(stderr, "Child n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
	//fprintf(stderr, "test: %d %d %d %d %d\n", block[n][AMR_CORN2], block[block[n][AMR_CORN2]][AMR_ACTIVE]);
}

void mpi_synch(int tag) {
	int i, n;
	int test1 = 0, test2 = 0;
	int gpu_block = 0;

	if (tag == 1) {
		MPI_Barrier(MPI_COMM_WORLD);
		gpu_block = 1;
	}
	else {
		/*for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
		if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
		break;
		}
		}*/
		for (i = 0; i < numtasks; i++) NODE_global[i] = 0;
		for (n = 0; n < n_active; n++) {
			//if (nstep%block[n_ord[n]][AMR_TIMELEVEL] == block[n_ord[n]][AMR_TIMELEVEL] - 1) gpu_block = 1;
			for (i = AMR_NBR1; i <= AMR_CORN12; i++) {
				if ((block[n_ord[n]][i] >= 0) && (block[block[n_ord[n]][i]][AMR_ACTIVE] == 1) && (nstep%block[block[n_ord[n]][i]][AMR_TIMELEVEL] == block[block[n_ord[n]][i]][AMR_TIMELEVEL] - 1) && (nstep%block[n_ord[n]][AMR_TIMELEVEL] == block[n_ord[n]][AMR_TIMELEVEL] - 1)) {
					NODE_global[block[block[n_ord[n]][i]][AMR_NODE]] = 10;
					gpu_block = 1;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) {
				if ((block[n_ord[n]][i] >= 0) && (block[block[n_ord[n]][i]][AMR_ACTIVE] == 1) && (nstep%block[block[n_ord[n]][i]][AMR_TIMELEVEL] == block[block[n_ord[n]][i]][AMR_TIMELEVEL] - 1) && (nstep%block[n_ord[n]][AMR_TIMELEVEL] == block[n_ord[n]][AMR_TIMELEVEL] - 1)) {
					NODE_global[block[block[n_ord[n]][i]][AMR_NODE]] = 10;
					gpu_block = 1;
				}
			}
		}
		for (i = 0; i < numtasks; i++) {
			if (NODE_global[i] == 10 && rank != i) {
				MPI_Isend(&test1, 1, MPI_INT, i, (4 * NB_LOCAL) % MPI_TAG_MAX, mpi_cartcomm, &req[i]);
				MPI_Irecv(&test2, 1, MPI_INT, i, (4 * NB_LOCAL) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[i]);
			}
		}
		for (i = 0; i < numtasks; i++) {
			if (NODE_global[i] == 10 && rank != i) {
				MPI_Wait(&req[i], &Statbound[0][0]);
				MPI_Wait(&request_timelevel[i], &Statbound[0][0]);
			}
		}
	}

	if (gpu_block == 1) {
		#if(GPU_ENABLED)
		for (n = gpu_offset; n < gpu_offset + N_GPU; n++) {
			#if(N_GPU>1)
			cudaSetDevice(n);
			#endif
			cudaDeviceSynchronize();
		}
		#endif
	}
}

void set_communicator(void) {
	int min_timelevel[8], i, n;
	for (i = 0; i <= log(AMR_MAXTIMELEVEL) / log(2); i++) {
		if (nstep > 2 * AMR_SWITCHTIMELEVEL) MPI_Comm_free(&row_comm[i]);

		min_timelevel[i] = rank + 1000000;
		for (n = 0; n < n_active; n++) {
			if (block[n_ord[n]][AMR_TIMELEVEL] <= pow(2, i)) min_timelevel[i] = 1;
		}
		MPI_Comm_split(mpi_cartcomm, min_timelevel[i], rank, &row_comm[i]);
	}
}

/*
void check_corners(void){
	int n;
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1){
			if ()

		}
	}
}*/