#include "decs_MPI.h"

/*Set the timelevel for a jet which is domain decomposed in the second dimension*/
void set_timelevel_jet(void){
	fprintf(stderr,"set_timelevel_jet is obselete and not anymore implemented! \n");
	exit(0);
}

/*Calculate for every block the timestep. This function should be node independent*/
void set_timelevel(void){
	int n;
	int i, j, z, l, ni, nj, nz;
	int task;
	int min_j[NB_1*32];
	MPI_Request req_local;
	ni = NB_1;
	nj = NB_2;
	nz = NB_3;
	
	const int i_max = log(AMR_MAXTIMELEVEL) / log(2);
	if (nstep > 0){
		#pragma omp parallel for schedule(dynamic,1) private(n,i)
		for (n = 0; n < n_active; n++){
			block[n_ord[n]][AMR_TIMELEVEL] = 1;
			for (i = i_max; i >= 0; i--){
				if (bdt[nl[n_ord[n]]][0] / dt >=1.0 * pow(2, i)){
					block[n_ord[n]][AMR_TIMELEVEL] = pow(2, i);
					break;
				}
			}
		}
	}

	//First make sure all nodes have the same information regarding the timestep
	//Send for every block (l,i,j,z) to block (l2,i,j2,z2) on other nodes using non-blocking send
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
			//#pragma omp parallel for schedule(dynamic,1) private(req_local, task)
			for (task = 0; task < numtasks; task++){
				if (task != rank){
					rc = MPI_Isend(&block[n_ord_total[n]][AMR_TIMELEVEL], 1, MPI_INT, task, (2 * NB_LOCAL + block[n_ord_total[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req_local);
					MPI_Request_free(&req_local);
				}
			}
		}
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			rc = MPI_Irecv(&(block[n_ord_total[n]][AMR_TIMELEVEL]), 1, MPI_INT, block[n_ord_total[n]][AMR_NODE], (2 * NB_LOCAL + block[n_ord_total[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
		}
	}

	//Receive from other nodes using blocking receive
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
		}
	}

	//Fixate the timestep around the pole
	for (l = 0; l < N_LEVELS_3D; l++){
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3, l);
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_ACTIVE] == 1){
				min_j[i] = 10000;
				if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 1 || block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 2 || block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_POLE] == 3){
					for (z = 0; z < NB_3; z++){
						min_j[i] = MY_MIN(block[AMR_coord_linear2(l, 0, i, 0, z)][AMR_TIMELEVEL], min_j[i]);
					}
					for (z = 0; z < NB_3; z++){
						block[AMR_coord_linear2(l, 0, i, 0, z)][AMR_TIMELEVEL] = min_j[i];
					}
				}
			}
		}
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, nj - 1, i, nj - 1, 0)][AMR_ACTIVE] == 1){
				min_j[i] = 10000;
				if (block[AMR_coord_linear2(l, nj - 1, i, nj - 1, 0)][AMR_POLE] == 1 || block[AMR_coord_linear2(l, nj - 1, i, nj - 1, 0)][AMR_POLE] == 2 || block[AMR_coord_linear2(l, nj - 1, i, nj - 1, 0)][AMR_POLE] == 3){
					for (z = 0; z < NB_3; z++){
						min_j[i] = MY_MIN(block[AMR_coord_linear2(l, nj - 1, i, nj - 1, z)][AMR_TIMELEVEL], min_j[i]);
					}
					for (z = 0; z < NB_3; z++){
						block[AMR_coord_linear2(l, nj - 1, i, nj - 1, z)][AMR_TIMELEVEL] = min_j[i];
					}
				}
			}
		}
	}

	//Create communicators for nodes which have a minimum (i) timelevel
	set_communicator();
	set_corners();
}

void set_communicator(void){
	int min_timelevel[8],i,n;
	for (i = 0; i <= log(AMR_MAXTIMELEVEL) / log(2); i++){
		if (nstep > 2 * AMR_SWITCHTIMELEVEL) MPI_Comm_free(&row_comm[i]);

		min_timelevel[i] = rank + 1000;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_TIMELEVEL] <= pow(2, i)) min_timelevel[i] = 1;
		}
		MPI_Comm_split(mpi_cartcomm, min_timelevel[i], rank, &row_comm[i]);
	}
}

void set_prestep(void){
	int n;

	#if(PRESTEP || PRESTEP2)
	int timelevel_min = AMR_MAXTIMELEVEL;
	int blocks_per_timestep = 0;
	int blocks_this_timestep = 0;

	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	//Find the minimum timelevel on this node
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_TIMELEVEL] < timelevel_min) timelevel_min = block[n_ord[n]][AMR_TIMELEVEL];
	}

	//Calculate the number of blocks you want to evolve simultaneously
	blocks_per_timestep = (count_node[0] - count_node[0] % (AMR_MAXTIMELEVEL / timelevel_min)) / (AMR_MAXTIMELEVEL / timelevel_min);
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep++;
	}

	//First make sure that the block required for prestepping in case of 2nd order time accuracy are preevolved
	#if(PRESTEP2)
	int i, j;
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && block[n_ord[n]][AMR_PRESTEP] == 0 && block[n_ord[n]][AMR_TIMELEVEL]>1){
			for (i = AMR_NBR1; i <= AMR_NBR6; i++){
				if (block[n_ord[n]][i] >= 0 && block[block[n_ord[n]][i]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep++;
					break;
				}
			}
			for (i = AMR_NBR1P; i <= AMR_NBR6P; i++){
				if (block[n_ord[n]][i] >= 0 && block[n_ord[n]][i]] >= 0 && block[block[n_ord[n]][i]]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep++;
					break;
				}
			}
			for (i = AMR_NBR1_3; i <= AMR_NBR6_8; i++){
				if (block[n_ord[n]][i] >= 0 && block[n_ord[n]][i]] >= 0 && block[block[n_ord[n]][i]]][AMR_ACTIVE] == 1 && block[block[n_ord[n]][i]]][AMR_TIMELEVEL] < block[n_ord[n]][AMR_TIMELEVEL]){
					block[n_ord[n]][AMR_PRESTEP] = 1;
					blocks_this_timestep++;
					break;
				}
			}
			if (block[n_ord[n]][AMR_POLE] > 0){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				blocks_this_timestep++;
			}
			if (block[n_ord[n]][AMR_PRESTEP] == 1) block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
		}
	}
	#endif

	//If you don't have sufficient blocks this timestep preevolve some blocks if available
	if ((nstep % timelevel_min) == timelevel_min - 1){
		for (n = 0; n < n_active; n++){
			if (blocks_this_timestep < blocks_per_timestep && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) != block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0 && (block[n_ord[n]][AMR_POLE] == 0)){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
				blocks_this_timestep++;
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
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_boundprim1(1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_boundprim2(1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (prestep_full[nl[n_ord[n]]] == 1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
		else if (prestep_half[nl[n_ord[n]]] == 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
	}

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (prestep_full[nl[n_ord[n]]] == 1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
		else if (prestep_half[nl[n_ord[n]]] == 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
	}

	if (N3 > 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (prestep_full[nl[n_ord[n]]] == 1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 1);
			else if (prestep_half[nl[n_ord[n]]] == 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 1);
		}
	}

	set_iprobe(0, &flag);
	do{
		//Store difference between evolved and required flux/electric field in temporary array
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			#if(N3G>0)
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	#endif

	set_iprobe(0, &flag);
	do{
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			E_rec1(E_corn, BufferE_1, n_ord[n], 1);
			E_rec2(E_corn, BufferE_1, n_ord[n], 1);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	#endif
	#endif
	//Then reset flux and electric fields to zero
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) GPU_cleanup_post(n_ord[n]);

	//Then insert flux differnce from temporary array in zeroed out flux and electric fields arrays
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		flux_rec1(F1, BufferF1_1, n_ord[n], 6);
		flux_rec2(F2, BufferF2_1, n_ord[n], 6);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 6);
		#endif
	}
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
	}
	#endif
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 6);
		E_rec2(E_corn, BufferE_1, n_ord[n], 6);
	}
	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 6);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 6);
	}
	#endif
	#endif

	//Evolve magnetic fields at boundary
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) GPU_consttransport3_post(dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	//Evolve conserved quantities at boundary using update fluxes and invert to primitive variables plus floor
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)) GPU_fixup_post(n_ord[n], dt*(double)block[n_ord[n]][AMR_TIMELEVEL]);
}

void set_corners(void){
	int n;
	int counter0, counter1, counter2, counter3;
	int counter0_1, counter1_1, counter2_1, counter3_1;
	int counter0_2, counter1_2, counter2_2, counter3_2;

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
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		//Corn 1
		block[n_ord_total[n]][AMR_CORN1D] = -10;
		block[n_ord_total[n]][AMR_CORN1D_1] = -10;
		block[n_ord_total[n]][AMR_CORN1D_2] = -10;

		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
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

			if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3){
				if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1){
					counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL];
					if (counter1 > counter0){
						block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_NBR1];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_REFINED] == 1){
					counter1_1 += 10000;
					counter1_2 += 10000;
					counter1 = 100000;
					block[n_ord_total[n]][AMR_CORN1D] = -2;
					counter1_1 -= block[block[n_ord_total[n]][AMR_NBR1_7]][AMR_TIMELEVEL];
					counter1_2 -= block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[n_ord_total[n]][AMR_NBR1_7];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[n_ord_total[n]][AMR_NBR1_8];
				}
			}

			if (block[n_ord_total[n]][AMR_NBR2] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1){
					counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL];
					if ((counter2 > counter1) && (counter2>counter0))block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_NBR2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_REFINED] == 1){
					counter2_1 += 10000;
					counter2_2 += 10000;
					counter2 = 100000;
					block[n_ord_total[n]][AMR_CORN1D] = -2;
					counter2_1 -= block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[n_ord_total[n]][AMR_NBR2_2]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[n_ord_total[n]][AMR_NBR2_1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[n_ord_total[n]][AMR_NBR2_2];
				}
			}

			if (block[n_ord_total[n]][AMR_CORN1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3){
				if (block[block[n_ord_total[n]][AMR_CORN1]][AMR_ACTIVE] == 1){
					counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN1]][AMR_TIMELEVEL];
					if ((counter3 > counter2) && (counter3>counter1) && (counter3>counter0)) block[n_ord_total[n]][AMR_CORN1D] = block[n_ord_total[n]][AMR_CORN1];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN1]][AMR_REFINED] == 1){
					counter3_1 += 10000;
					counter3_2 += 10000;
					counter3 = 100000;
					block[n_ord_total[n]][AMR_CORN1D] = -2;
					counter3_1 -= block[block[n_ord_total[n]][AMR_CORN1_1]][AMR_TIMELEVEL];
					counter3_2 -= block[block[n_ord_total[n]][AMR_CORN1_2]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN1D_1] =block[n_ord_total[n]][AMR_CORN1_1];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN1D_2] =block[n_ord_total[n]][AMR_CORN1_2];
				}
			}
		}
	}
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		block[n_ord_total[n]][AMR_CORN2D] = -10;
		block[n_ord_total[n]][AMR_CORN2D_1] = -10;
		block[n_ord_total[n]][AMR_CORN2D_2] = -10;
		block[n_ord_total[n]][AMR_CORN3D] = -10;
		block[n_ord_total[n]][AMR_CORN3D_1] = -10;
		block[n_ord_total[n]][AMR_CORN3D_2] = -10;
		block[n_ord_total[n]][AMR_CORN4D] = -10;
		block[n_ord_total[n]][AMR_CORN4D_1] = -10;
		block[n_ord_total[n]][AMR_CORN4D_2] = -10;

		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
			//Corn 2
			if (block[n_ord_total[n]][AMR_CORN2] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN2D_1] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D_1];
					block[n_ord_total[n]][AMR_CORN2D_2] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN2D] = -2;
					block[n_ord_total[n]][AMR_CORN2D_1] = block[block[n_ord_total[n]][AMR_NBR3_5]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN2D_2] = block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN1D];
				}
				else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == REF_1 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN1D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == REF_1 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2))
				{
					block[n_ord_total[n]][AMR_CORN2D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN2D] = block[n_ord_total[n]][AMR_NBR2];
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN2D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN1D]; //good
				}
			}

			//Corn 3
			if (block[n_ord_total[n]][AMR_CORN3] >= 0){
				if (block[block[n_ord_total[n]][AMR_CORN3]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN3D_1] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D_1];
					block[n_ord_total[n]][AMR_CORN3D_2] = block[block[n_ord_total[n]][AMR_CORN3]][AMR_CORN1D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN3]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN3D] = -2;
					block[n_ord_total[n]][AMR_CORN3D_1] = block[block[n_ord_total[n]][AMR_CORN3_1]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN3D_2] = block[block[n_ord_total[n]][AMR_CORN3_2]][AMR_CORN1D];
				}
				else if (block[n_ord_total[n]][AMR_CORN3P] >= 0 && block[block[n_ord_total[n]][AMR_CORN3P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN3D] = block[block[n_ord_total[n]][AMR_CORN3P]][AMR_CORN1D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_CORN3P] >= 0 && block[block[n_ord_total[n]][AMR_CORN3P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2))
				{
					block[n_ord_total[n]][AMR_CORN3D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR3]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN3D] = block[n_ord_total[n]][AMR_NBR3];//good
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR4]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN3D] = block[n_ord_total[n]][AMR_NBR4];//good
				}
			}

			//Corn 4
			if (block[n_ord_total[n]][AMR_CORN4] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN4D_1] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D_1];
					block[n_ord_total[n]][AMR_CORN4D_2] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN4D] = -2;
					block[n_ord_total[n]][AMR_CORN4D_1] = block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN4D_2] = block[block[n_ord_total[n]][AMR_NBR4_6]][AMR_CORN1D];
				}
				else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == 0)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN4D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN1D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == 0))
				{
					block[n_ord_total[n]][AMR_CORN4D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN1D]; //good
					if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN4D] = block[n_ord_total[n]][AMR_NBR1]; //good
				}
			}
			if (block[n_ord_total[n]][AMR_NBR4] < 0){
				block[n_ord_total[n]][AMR_CORN4D] = -100;
				block[n_ord_total[n]][AMR_CORN4D_1] = -100;
				block[n_ord_total[n]][AMR_CORN4D_2] = -100;
				block[n_ord_total[n]][AMR_CORN3D] = -100;
				block[n_ord_total[n]][AMR_CORN3D_1] = -100;
				block[n_ord_total[n]][AMR_CORN3D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR2] < 0){
				block[n_ord_total[n]][AMR_CORN1D] = -100;
				block[n_ord_total[n]][AMR_CORN1D_1] = -100;
				block[n_ord_total[n]][AMR_CORN1D_2] = -100;
				block[n_ord_total[n]][AMR_CORN2D] = -100;
				block[n_ord_total[n]][AMR_CORN2D_1] = -100;
				block[n_ord_total[n]][AMR_CORN2D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR1] < 0 || block[n_ord_total[n]][AMR_POLE] == 1 || block[n_ord_total[n]][AMR_POLE] == 3){
				block[n_ord_total[n]][AMR_CORN1D] = -100;
				block[n_ord_total[n]][AMR_CORN1D_1] = -100;
				block[n_ord_total[n]][AMR_CORN1D_2] = -100;
				block[n_ord_total[n]][AMR_CORN4D] = -100;
				block[n_ord_total[n]][AMR_CORN4D_1] = -100;
				block[n_ord_total[n]][AMR_CORN4D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR3] < 0 || block[n_ord_total[n]][AMR_POLE] == 2 || block[n_ord_total[n]][AMR_POLE] == 3){
				block[n_ord_total[n]][AMR_CORN3D] = -100;
				block[n_ord_total[n]][AMR_CORN3D_1] = -100;
				block[n_ord_total[n]][AMR_CORN3D_2] = -100;
				block[n_ord_total[n]][AMR_CORN2D] = -100;
				block[n_ord_total[n]][AMR_CORN2D_1] = -100;
				block[n_ord_total[n]][AMR_CORN2D_2] = -100;
			}
		}
	}
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		//Corn 5
		block[n_ord_total[n]][AMR_CORN5D] = -10;
		block[n_ord_total[n]][AMR_CORN5D_1] = -10;
		block[n_ord_total[n]][AMR_CORN5D_2] = -10;
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
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

			if (block[n_ord_total[n]][AMR_NBR6] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1){
					counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL];
					if (counter1 > counter0){
						block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_NBR6];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_REFINED] == 1){
					counter1_1 += 10000;
					counter1_2 += 10000;
					counter1 = 100000;
					block[n_ord_total[n]][AMR_CORN5D] = -2;
					counter1_1 -= block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_TIMELEVEL];
					counter1_2 -= block[block[n_ord_total[n]][AMR_NBR6_8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[n_ord_total[n]][AMR_NBR6_6];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[n_ord_total[n]][AMR_NBR6_8];
				}
			}

			if (block[n_ord_total[n]][AMR_NBR2] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1){
					counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL];
					if ((counter2 > counter1) && (counter2>counter0))block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_NBR2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_REFINED] == 1){
					counter2_1 += 10000;
					counter2_2 += 10000;
					counter2 = 100000;
					block[n_ord_total[n]][AMR_CORN5D] = -2;
					counter2_1 -= block[block[n_ord_total[n]][AMR_NBR2_1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[n_ord_total[n]][AMR_NBR2_3]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[n_ord_total[n]][AMR_NBR2_1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[n_ord_total[n]][AMR_NBR2_3];
				}
			}

			if (block[n_ord_total[n]][AMR_CORN5] >= 0){
				if (block[block[n_ord_total[n]][AMR_CORN5]][AMR_ACTIVE] == 1){
					counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN5]][AMR_TIMELEVEL];
					if ((counter3 > counter2) && (counter3>counter1) && (counter3>counter0)) block[n_ord_total[n]][AMR_CORN5D] = block[n_ord_total[n]][AMR_CORN5];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN5]][AMR_REFINED] == 1){
					counter3_1 += 10000;
					counter3_2 += 10000;
					counter3 = 100000;
					block[n_ord_total[n]][AMR_CORN5D] = -2;
					counter3_1 -= block[block[n_ord_total[n]][AMR_CORN5_1]][AMR_TIMELEVEL];
					counter3_2 -= block[block[n_ord_total[n]][AMR_CORN5_2]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN5D_1] =block[n_ord_total[n]][AMR_CORN5_1];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN5D_2] =block[n_ord_total[n]][AMR_CORN5_2];
				}
			}
		}
	}

	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		block[n_ord_total[n]][AMR_CORN6D] = -10;
		block[n_ord_total[n]][AMR_CORN6D_1] = -10;
		block[n_ord_total[n]][AMR_CORN6D_2] = -10;
		block[n_ord_total[n]][AMR_CORN7D] = -10;
		block[n_ord_total[n]][AMR_CORN7D_1] = -10;
		block[n_ord_total[n]][AMR_CORN7D_2] = -10;
		block[n_ord_total[n]][AMR_CORN8D] = -10;
		block[n_ord_total[n]][AMR_CORN8D_1] = -10;
		block[n_ord_total[n]][AMR_CORN8D_2] = -10;
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
			//Corn 6
			if (block[n_ord_total[n]][AMR_CORN6] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN6D_1] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D_1];
					block[n_ord_total[n]][AMR_CORN6D_2] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN6D] = -2;
					block[n_ord_total[n]][AMR_CORN6D_1] = block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN6D_2] = block[block[n_ord_total[n]][AMR_NBR5_7]][AMR_CORN5D];
				}
				else if (block[n_ord_total[n]][AMR_NBR5P] >= 0 && block[block[n_ord_total[n]][AMR_NBR5P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == REF_1 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5P]][AMR_CORN5D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR5P] >= 0 && block[block[n_ord_total[n]][AMR_NBR5P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == REF_1 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3))
				{
					block[n_ord_total[n]][AMR_CORN6D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR2]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR2]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN6D] = block[n_ord_total[n]][AMR_NBR2];
					if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN6D] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CORN5D]; //good
				}
			}

			//Corn 7
			if (block[n_ord_total[n]][AMR_CORN7] >= 0){
				if (block[block[n_ord_total[n]][AMR_CORN7]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN7D_1] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D_1];
					block[n_ord_total[n]][AMR_CORN7D_2] = block[block[n_ord_total[n]][AMR_CORN7]][AMR_CORN5D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN7]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN7D] = -2;
					block[n_ord_total[n]][AMR_CORN7D_1] = block[block[n_ord_total[n]][AMR_CORN7_1]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN7D_2] = block[block[n_ord_total[n]][AMR_CORN7_2]][AMR_CORN5D];
				}
				else if (block[n_ord_total[n]][AMR_CORN7P] >= 0 && block[block[n_ord_total[n]][AMR_CORN7P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN7D] = block[block[n_ord_total[n]][AMR_CORN7P]][AMR_CORN5D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_CORN7P] >= 0 && block[block[n_ord_total[n]][AMR_CORN7P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3))
				{
					block[n_ord_total[n]][AMR_CORN7D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN7D] = block[n_ord_total[n]][AMR_NBR5];//good
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR4]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN7D] = block[n_ord_total[n]][AMR_NBR4];//good
				}
			}

			//Corn 8
			if (block[n_ord_total[n]][AMR_CORN8] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN8D_1] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D_1];
					block[n_ord_total[n]][AMR_CORN8D_2] = block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN8D] = -2;
					block[n_ord_total[n]][AMR_CORN8D_1] = block[block[n_ord_total[n]][AMR_NBR4_5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN8D_2] = block[block[n_ord_total[n]][AMR_NBR4_7]][AMR_CORN5D];
				}
				else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN8D] = block[block[n_ord_total[n]][AMR_NBR4P]][AMR_CORN5D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR4P] >= 0 && block[block[n_ord_total[n]][AMR_NBR4P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD1] % (1 + REF_1) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0))
				{
					block[n_ord_total[n]][AMR_CORN8D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR4]][AMR_CORN5D]; //good
					if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN8D] = block[n_ord_total[n]][AMR_NBR6]; //good
				}
			}

			if (block[n_ord_total[n]][AMR_NBR4] < 0){
				block[n_ord_total[n]][AMR_CORN8D] = -100;
				block[n_ord_total[n]][AMR_CORN8D_1] = -100;
				block[n_ord_total[n]][AMR_CORN8D_2] = -100;
				block[n_ord_total[n]][AMR_CORN7D] = -100;
				block[n_ord_total[n]][AMR_CORN7D_1] = -100;
				block[n_ord_total[n]][AMR_CORN7D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR2] < 0){
				block[n_ord_total[n]][AMR_CORN5D] = -100;
				block[n_ord_total[n]][AMR_CORN5D_1] = -100;
				block[n_ord_total[n]][AMR_CORN5D_2] = -100;
				block[n_ord_total[n]][AMR_CORN6D] = -100;
				block[n_ord_total[n]][AMR_CORN6D_1] = -100;
				block[n_ord_total[n]][AMR_CORN6D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR6] < 0){
				block[n_ord_total[n]][AMR_CORN5D] = -100;
				block[n_ord_total[n]][AMR_CORN5D_1] = -100;
				block[n_ord_total[n]][AMR_CORN5D_2] = -100;
				block[n_ord_total[n]][AMR_CORN8D] = -100;
				block[n_ord_total[n]][AMR_CORN8D_1] = -100;
				block[n_ord_total[n]][AMR_CORN8D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR5] < 0){
				block[n_ord_total[n]][AMR_CORN7D] = -100;
				block[n_ord_total[n]][AMR_CORN7D_1] = -100;
				block[n_ord_total[n]][AMR_CORN7D_2] = -100;
				block[n_ord_total[n]][AMR_CORN6D] = -100;
				block[n_ord_total[n]][AMR_CORN6D_1] = -100;
				block[n_ord_total[n]][AMR_CORN6D_2] = -100;
			}
		}
	}
	//Set the most important corner value of the electric field to break the degeneracy of E-fields at each corner
	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		//Corn 9
		block[n_ord_total[n]][AMR_CORN9D] = -10;
		block[n_ord_total[n]][AMR_CORN9D_1] = -10;
		block[n_ord_total[n]][AMR_CORN9D_2] = -10;
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
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

			if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3){
				if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1){
					counter1 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL];
					if (counter1 > counter0){
						block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_NBR1];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_REFINED] == 1){
					counter1_1 += 10000;
					counter1_2 += 10000;
					counter1 = 100000;
					block[n_ord_total[n]][AMR_CORN9D] = -2;
					counter1_1 -= block[block[n_ord_total[n]][AMR_NBR1_4]][AMR_TIMELEVEL];
					counter1_2 -= block[block[n_ord_total[n]][AMR_NBR1_8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[n_ord_total[n]][AMR_NBR1_4];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[n_ord_total[n]][AMR_NBR1_8];
				}
			}

			if (block[n_ord_total[n]][AMR_NBR5] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1){
					counter2 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL];
					if ((counter2 > counter1) && (counter2>counter0))block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_NBR5];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_REFINED] == 1){
					counter2_1 += 10000;
					counter2_2 += 10000;
					counter2 = 100000;
					block[n_ord_total[n]][AMR_CORN9D] = -2;
					counter2_1 -= block[block[n_ord_total[n]][AMR_NBR5_1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[n_ord_total[n]][AMR_NBR5_5]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[n_ord_total[n]][AMR_NBR5_1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[n_ord_total[n]][AMR_NBR5_5];
				}
			}

			if (block[n_ord_total[n]][AMR_CORN9] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 3){
				if (block[block[n_ord_total[n]][AMR_CORN9]][AMR_ACTIVE] == 1){
					counter3 = AMR_MAXTIMELEVEL - block[block[n_ord_total[n]][AMR_CORN9]][AMR_TIMELEVEL];
					if ((counter3 > counter2) && (counter3>counter1) && (counter3>counter0)) block[n_ord_total[n]][AMR_CORN9D] = block[n_ord_total[n]][AMR_CORN9];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN9]][AMR_REFINED] == 1){
					counter3_1 += 10000;
					counter3_2 += 10000;
					counter3 = 100000;
					block[n_ord_total[n]][AMR_CORN9D] = -2;
					counter3_1 -= block[block[n_ord_total[n]][AMR_CORN9_1]][AMR_TIMELEVEL];
					counter3_2 -= block[block[n_ord_total[n]][AMR_CORN9_2]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN9D_1] =block[n_ord_total[n]][AMR_CORN9_1];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN9D_2] =block[n_ord_total[n]][AMR_CORN9_2];
				}
			}
		}
	}

	#pragma omp parallel for schedule(dynamic,1) private(n,counter0, counter1, counter2, counter3,counter0_1, counter1_1, counter2_1, counter3_1, counter0_2, counter1_2, counter2_2, counter3_2)
	for (n = 0; n < n_active_total; n++){
		block[n_ord_total[n]][AMR_CORN10D] = -10;
		block[n_ord_total[n]][AMR_CORN10D_1] = -10;
		block[n_ord_total[n]][AMR_CORN10D_2] = -10;
		block[n_ord_total[n]][AMR_CORN11D] = -10;
		block[n_ord_total[n]][AMR_CORN11D_1] = -10;
		block[n_ord_total[n]][AMR_CORN11D_2] = -10;
		block[n_ord_total[n]][AMR_CORN12D] = -10;
		block[n_ord_total[n]][AMR_CORN12D_1] = -10;
		block[n_ord_total[n]][AMR_CORN12D_2] = -10;
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1){
			//Corn 10
			if (block[n_ord_total[n]][AMR_CORN10] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN10D_1] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D_1];
					block[n_ord_total[n]][AMR_CORN10D_2] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN10D] = -2;
					block[n_ord_total[n]][AMR_CORN10D_1] = block[block[n_ord_total[n]][AMR_NBR3_2]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN10D_2] = block[block[n_ord_total[n]][AMR_NBR3_6]][AMR_CORN9D];
				}
				else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3P]][AMR_CORN9D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR3P] >= 0 && block[block[n_ord_total[n]][AMR_NBR3P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == REF_3))
				{
					block[n_ord_total[n]][AMR_CORN10D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR5]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL]) block[n_ord_total[n]][AMR_CORN10D] = block[n_ord_total[n]][AMR_NBR5];
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1) block[n_ord_total[n]][AMR_CORN10D] = block[block[n_ord_total[n]][AMR_NBR3]][AMR_CORN9D]; //good
				}
			}

			//Corn 11
			if (block[n_ord_total[n]][AMR_CORN11] >= 0){
				if (block[block[n_ord_total[n]][AMR_CORN11]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN11D_1] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D_1];
					block[n_ord_total[n]][AMR_CORN11D_2] = block[block[n_ord_total[n]][AMR_CORN11]][AMR_CORN9D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN11]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN11D] = -2;
					block[n_ord_total[n]][AMR_CORN11D_1] = block[block[n_ord_total[n]][AMR_CORN11_1]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN11D_2] = block[block[n_ord_total[n]][AMR_CORN11_2]][AMR_CORN9D];
				}
				else if (block[n_ord_total[n]][AMR_CORN11P] >= 0 && block[block[n_ord_total[n]][AMR_CORN11P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN11D] = block[block[n_ord_total[n]][AMR_CORN11P]][AMR_CORN9D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_CORN11P] >= 0 && block[block[n_ord_total[n]][AMR_CORN11P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == REF_2 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0))
				{
					block[n_ord_total[n]][AMR_CORN11D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR3]][AMR_TIMELEVEL] <= block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN11D] = block[n_ord_total[n]][AMR_NBR3];//good
					if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR6]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN11D] = block[n_ord_total[n]][AMR_NBR6];//good
				}
			}

			//Corn 12
			if (block[n_ord_total[n]][AMR_CORN12] >= 0){
				if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1){
					block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN12D_1] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D_1];
					block[n_ord_total[n]][AMR_CORN12D_2] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D_2];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_REFINED] == 1){
					block[n_ord_total[n]][AMR_CORN12D] = -2;
					block[n_ord_total[n]][AMR_CORN12D_1] = block[block[n_ord_total[n]][AMR_NBR6_2]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN12D_2] = block[block[n_ord_total[n]][AMR_NBR6_6]][AMR_CORN9D];
				}
				else if (block[n_ord_total[n]][AMR_NBR6P] >= 0 && block[block[n_ord_total[n]][AMR_NBR6P]][AMR_ACTIVE] == 1 && (block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0)){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN12D] = block[block[n_ord_total[n]][AMR_NBR6P]][AMR_CORN9D_2];
					}
				}
				else if (block[n_ord_total[n]][AMR_NBR6P] >= 0 && block[block[n_ord_total[n]][AMR_NBR6P]][AMR_ACTIVE] == 1 && !(block[n_ord_total[n]][AMR_COORD2] % (1 + REF_2) == 0 && block[n_ord_total[n]][AMR_COORD3] % (1 + REF_3) == 0))
				{
					block[n_ord_total[n]][AMR_CORN12D] = n_ord_total[n];
					if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_ACTIVE] == 1) block[block[n_ord_total[n]][AMR_NBR6]][AMR_CORN9D]; //good
					if (block[block[n_ord_total[n]][AMR_NBR1]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_NBR1]][AMR_TIMELEVEL] < block[n_ord_total[n]][AMR_TIMELEVEL])block[n_ord_total[n]][AMR_CORN12D] = block[n_ord_total[n]][AMR_NBR1]; //good
				}
			}
			if (block[n_ord_total[n]][AMR_NBR6] < 0){
				block[n_ord_total[n]][AMR_CORN12D] = -100;
				block[n_ord_total[n]][AMR_CORN12D_1] = -100;
				block[n_ord_total[n]][AMR_CORN12D_2] = -100;
				block[n_ord_total[n]][AMR_CORN11D] = -100;
				block[n_ord_total[n]][AMR_CORN11D_1] = -100;
				block[n_ord_total[n]][AMR_CORN11D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR5] < 0){
				block[n_ord_total[n]][AMR_CORN9D] = -100;
				block[n_ord_total[n]][AMR_CORN9D_1] = -100;
				block[n_ord_total[n]][AMR_CORN9D_2] = -100;
				block[n_ord_total[n]][AMR_CORN10D] = -100;
				block[n_ord_total[n]][AMR_CORN10D_1] = -100;
				block[n_ord_total[n]][AMR_CORN10D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR1] < 0 || block[n_ord_total[n]][AMR_POLE] == 1 || block[n_ord_total[n]][AMR_POLE] == 3){
				block[n_ord_total[n]][AMR_CORN9D] = -100;
				block[n_ord_total[n]][AMR_CORN9D_1] = -100;
				block[n_ord_total[n]][AMR_CORN9D_2] = -100;
				block[n_ord_total[n]][AMR_CORN12D] = -100;
				block[n_ord_total[n]][AMR_CORN12D_1] = -100;
				block[n_ord_total[n]][AMR_CORN12D_2] = -100;
			}
			if (block[n_ord_total[n]][AMR_NBR3] < 0 || block[n_ord_total[n]][AMR_POLE] == 2 || block[n_ord_total[n]][AMR_POLE] == 3){
				block[n_ord_total[n]][AMR_CORN11D] = -100;
				block[n_ord_total[n]][AMR_CORN11D_1] = -100;
				block[n_ord_total[n]][AMR_CORN11D_2] = -100;
				block[n_ord_total[n]][AMR_CORN10D] = -100;
				block[n_ord_total[n]][AMR_CORN10D_1] = -100;
				block[n_ord_total[n]][AMR_CORN10D_2] = -100;
			}
		}
	}
#endif
}
