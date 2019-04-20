#include "decs_MPI.h"

#if(STAGGERED)
void const_transport1(double(*restrict pb[NB_LOCAL])[NPR], int n){
	int i, j, z, k, ind0;
	double E_cent[NDIM];
	struct of_state q;
	struct of_geom geom;

	#pragma omp parallel shared(n,n_ord,n_active,E_corn, F1, F2, F3, dx,pb, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0, E_cent, geom, q)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)*(BS_3+2*D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1 - D1, (N1_GPU_offset[n] + BS_1)*D1, N2_GPU_offset[n] * D2 - D2, (N2_GPU_offset[n] + BS_2)*D2, N3_GPU_offset[n] * D3 - D3, (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);

			//calculate the corner values of the electric field by averaging the Godunov fluxes, see formula 7 balsara&spicer
			#if(N3G>0)
			E_corn[nl[n]][ind0][1] = 0.25*(F3[nl[n]][ind0][B2] + F3[nl[n]][index_3D(n, i, j - D2, z)][B2] - F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			E_corn[nl[n]][ind0][2] = 0.25*(F1[nl[n]][ind0][B3] + F1[nl[n]][index_3D(n, i, j, z - D3)][B3] - F3[nl[n]][ind0][B1] - F3[nl[n]][index_3D(n, i - D1, j, z)][B1]);
			#endif
			E_corn[nl[n]][ind0][3] = 0.25*(F2[nl[n]][ind0][B1] + F2[nl[n]][index_3D(n, i - D1, j, z)][B1] - F1[nl[n]][ind0][B2] - F1[nl[n]][index_3D(n, i, j - D2, z)][B2]);

			get_geometry(n, i, j, z, CENT, &geom);
			get_state(pb[nl[n]][ind0], &geom, &q);

			//calculate the cell center values of the E-field
			#if(N3G>0)
			E_cent[1] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
			E_cent[2] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
			#endif
			E_cent[3] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);

			//upwind the electric field based on transverse gradients conform gardiner&stone 2005/2015, not yet tested
			#if(N3G>0)
			dE[nl[n]][ind0][LEFT][1][2] = (E_cent[1] + F2[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][1][3] = (E_cent[1] - F3[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][2][1] = (E_cent[2] - F1[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][2][3] = (E_cent[2] + F3[nl[n]][ind0][B1]);
			#endif
			dE[nl[n]][ind0][LEFT][3][1] = (E_cent[3] + F1[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][3][2] = (E_cent[3] - F2[nl[n]][ind0][B1]);

			#if(N3G>0)
			dE[nl[n]][ind0][RIGHT][1][2] = (-F2[nl[n]][index_3D(n, i, j + D2, z)][B3] - E_cent[1]);
			dE[nl[n]][ind0][RIGHT][1][3] = (F3[nl[n]][index_3D(n, i, j, z + D3)][B2] - E_cent[1]);
			dE[nl[n]][ind0][RIGHT][2][1] = (F1[nl[n]][index_3D(n, i + D1, j, z)][B3] - E_cent[2]);
			dE[nl[n]][ind0][RIGHT][2][3] = (-F3[nl[n]][index_3D(n, i, j, z + D3)][B1] - E_cent[2]);
			#endif
			dE[nl[n]][ind0][RIGHT][3][1] = (-F1[nl[n]][index_3D(n, i + D1, j, z)][B2] - E_cent[3]);
			dE[nl[n]][ind0][RIGHT][3][2] = (F2[nl[n]][index_3D(n, i, j + D2, z)][B1] - E_cent[3]);
		}

		#pragma omp for collapse(2) schedule(static,(BS_1+D1)*(BS_2+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1, (N1_GPU_offset[n] + BS_1)*D1, N2_GPU_offset[n] * D2, (N2_GPU_offset[n] + BS_2)*D2, N3_GPU_offset[n] * D3, (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);

			E_corn[nl[n]][ind0][1] = 0.25*((-F2[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO]>0.0)))
				+ (-F2[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO]>0.0)))
				+ (F3[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO]>0.0)))
				+ (F3[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO]>0.0))));
			E_corn[nl[n]][ind0][2] = 0.25*((-F3[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] > 0.0)))
				+ (-F3[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (F1[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (F1[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] > 0.0))));
			E_corn[nl[n]][ind0][3] = 0.25*((F2[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] > 0.0)))
				+ (F2[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (-F1[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (-F1[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] > 0.0))));

			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][1] = 0.5*(-F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][3] = 0.0;
		}
	}
}

void const_transport_bound(void){
	int n, flag;
	gpu = 0;

	#if(TRANS_BOUND)
	E_average();
	#endif

	set_iprobe(0, &flag);
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, Bufferdq_1, n_ord[n], 2);
	#endif
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send1(E_corn, Bufferdq_1, n_ord[n]);
	do{
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, Bufferdq_1, n_ord[n], 1);
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, Bufferdq_1, n_ord[n], 2);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send2(E_corn, Bufferdq_1, n_ord[n]);
	do{
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, Bufferdq_1, n_ord[n], 1);
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, Bufferdq_1, n_ord[n], 2);
	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send3(E_corn, Bufferdq_1, n_ord[n]);
	do{
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, Bufferdq_1, n_ord[n], 1);
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, Bufferdq_1, n_ord[n], 2);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, Bufferdq_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, Bufferdq_1, n_ord[n], 2);
	#endif	
	#endif
}

void E_average(void){
	int n, n1, n2, i, j, z, z2,z3, tag, k, ind0, z_max, number, number_rec, u;
	MPI_Request req_local;
	int l, ni, nj, nz;

	//Read in average value of E1 at pole for every block on node
	for (n = 0; n < n_active; n++) if (prestep_full[nl[n_ord[n]]] == 1 || prestep_half[nl[n_ord[n]]] == 1){
		read_E_avg(E_avg1[block[n_ord[n]][AMR_LEVEL]], E_avg2[block[n_ord[n]][AMR_LEVEL]], n_ord[n]);
	}

	//If block is not on node send the data to other node over MPI for positive pole
	for (l = 0; l < N_LEVELS_3D; l++){
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3*(!DEREFINE_POLE), l);
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_ACTIVE] == 1) {
				//Which nodes have an active block around a slice in phi for a given i
				if ((nstep % (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL] - 1 && !PRESTEP2) || (PRESTEP2 && nstep % (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL]) == 0)) {
					for (z = 0; z < nz; z++) {
						number = AMR_coord_linear2(l, 0, i, 0, z);
						if (block[number][AMR_NODE] == rank) {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, 0, i, 0, z2)][AMR_NODE];
								if (u != rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, 0, i, 0, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										rc = MPI_Isend(&E_avg1[l][i*nz + z][0], (BS_1 + 2 * N1G), MPI_DOUBLE, u, (8 * NB_LOCAL + block[number][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req_local);
										MPI_Request_free(&req_local);
									}
								}
							}
						}
						else {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, 0, i, 0, z2)][AMR_NODE];
								if (u == rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, 0, i, 0, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										rc = MPI_Irecv(&E_avg1[l][i*nz + z][0], (BS_1 + 2 * N1G), MPI_DOUBLE, block[number][AMR_NODE], (8 * NB_LOCAL + block[number][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req_local1[l][i*nz + z]);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	//If block is not on node send the data to other node over MPI for negative pole
	for (l = 0; l < N_LEVELS_3D; l++){
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3*(!DEREFINE_POLE), l);
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_ACTIVE] == 1){
				if ((nstep % (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL] - 1 && !PRESTEP2) || (PRESTEP2 && nstep % (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL]) == 0)){
					for (z = 0; z < nz; z++) {
						number = AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z);
						if (block[number][AMR_NODE] == rank) {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z2)][AMR_NODE];
								if (u != rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										rc = MPI_Isend(&E_avg2[l][i*nz + z][0], (BS_1 + 2 * N1G), MPI_DOUBLE, u, (9 * NB_LOCAL + block[number][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req_local);
										MPI_Request_free(&req_local);
									}
								}
							}
						}
						else {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z2)][AMR_NODE];
								if (u == rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										rc = MPI_Irecv(&E_avg2[l][i*nz + z][0], (BS_1 + 2 * N1G), MPI_DOUBLE, block[number][AMR_NODE], (9 * NB_LOCAL + block[number][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req_local2[l][i*nz + z]);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	//Which nodes have an active block around a slice in phi for a given i
	for (l = 0; l < N_LEVELS_3D; l++){
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3*(!DEREFINE_POLE), l);
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_ACTIVE] == 1){
				if ((nstep % (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL] - 1 && !PRESTEP2) || (PRESTEP2 && nstep % (block[AMR_coord_linear2(l, 0, i, 0, 0)][AMR_TIMELEVEL]) == 0)){
					for (z = 0; z < nz; z++) {
						number = AMR_coord_linear2(l, 0, i,0, z);
						if (block[number][AMR_NODE] != rank) {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, 0, i, 0, z2)][AMR_NODE];
								if (u == rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, 0, i, 0, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										MPI_Wait(&req_local1[l][i*nz + z], &Statbound[0][491]);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	for (l = 0; l < N_LEVELS_3D; l++){
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3*(!DEREFINE_POLE), l);
		for (i = 0; i < ni; i++){
			if (block[AMR_coord_linear2(l, NB_2-1, i, nj - 1, 0)][AMR_ACTIVE] == 1){
				if ((nstep % (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL] - 1 && !PRESTEP2) || (PRESTEP2 && nstep % (block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, 0)][AMR_TIMELEVEL]) == 0)){
					for (z = 0; z < nz; z++) {
						number = AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z);
						if (block[number][AMR_NODE] != rank) {
							for (z2 = 0; z2 < nz; z2++) {
								u = block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z2)][AMR_NODE];
								if (u == rank) {
									tag = 1;
									for (z3 = z2 - 1; z3 >= 0; z3--) {
										if (u == block[AMR_coord_linear2(l, NB_2 - 1, i, nj - 1, z3)][AMR_NODE]) tag = 0;
									}
									if (tag == 1) {
										MPI_Wait(&req_local2[l][i*nz + z], &Statbound[0][491]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	//Average the first component of the E_field for both poles
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 || prestep_half[nl[n_ord[n]]] == 1){
		nz = NB_3 * pow(1 + REF_3, block[n_ord[n]][AMR_LEVEL3]);
		if (block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3){
			z_max = nz;
			for (z = 0; z < z_max; z++){
				number = AMR_coord_linear2(block[n_ord[n]][AMR_LEVEL], 0, block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
				for (i = 0; i < BS_1 + D1; i++){
					if (z == 0)E_avg1_new[block[n_ord[n]][AMR_LEVEL]][block[n_ord[n]][AMR_COORD1] * nz + block[n_ord[n]][AMR_COORD3]][i] = E_avg1[block[n_ord[n]][AMR_LEVEL]][block[number][AMR_COORD1] * nz + block[number][AMR_COORD3]][i] / ((double)z_max);
					else E_avg1_new[block[n_ord[n]][AMR_LEVEL]][block[n_ord[n]][AMR_COORD1] * nz + block[n_ord[n]][AMR_COORD3]][i] += E_avg1[block[n_ord[n]][AMR_LEVEL]][block[number][AMR_COORD1] * nz + block[number][AMR_COORD3]][i] / ((double)z_max);
				}
			}
		}

		if (block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			z_max = nz;
			for (z = 0; z < z_max; z++){
				number = AMR_coord_linear2(block[n_ord[n]][AMR_LEVEL], NB_2 - 1, block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
				for (i = 0; i < BS_1 + D1; i++){
					if (z == 0)E_avg2_new[block[n_ord[n]][AMR_LEVEL]][block[n_ord[n]][AMR_COORD1] * nz + block[n_ord[n]][AMR_COORD3]][i] = E_avg2[block[n_ord[n]][AMR_LEVEL]][block[number][AMR_COORD1] * nz + block[number][AMR_COORD3]][i] / ((double)z_max);
					else E_avg2_new[block[n_ord[n]][AMR_LEVEL]][block[n_ord[n]][AMR_COORD1] * nz + block[n_ord[n]][AMR_COORD3]][i] += E_avg2[block[n_ord[n]][AMR_LEVEL]][block[number][AMR_COORD1] * nz + block[number][AMR_COORD3]][i] / ((double)z_max);
				}
			}
		}
	}

	//Write average value of E1 at pole for every block on node
	for (n = 0; n < n_active; n++) if (prestep_full[nl[n_ord[n]]]==1 || prestep_half[nl[n_ord[n]]]==1){
		write_E_avg(E_avg1_new[block[n_ord[n]][AMR_LEVEL]], E_avg2_new[block[n_ord[n]][AMR_LEVEL]], n_ord[n]);
	}
}

void read_E_avg(double(*E_avg1)[BS_1 + 2 * N1G], double(*E_avg2)[BS_1 + 2 * N1G], int n){
	int i, i1, i2, z, z1, z2, isize, zsize, nz;
	//double ph;
	i1 = 0;
	i2 = BS_1 + N1G;
	z1 = 0;
	z2 = BS_3 + N3G;
	isize = (BS_1 + N1G);
	zsize = (BS_3 + N3G);
	nz = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]);

	#if (N_GPU>1)
	cudaSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		pack_send2_E(n, n, i1, i2, 0, D2, z1, z2, isize, zsize, send1_fine, E_corn, &(BufferE_1[nl[n]]), &(Buffersend1fine[nl[n]]), &(boundevent[nl[n]][399]));
		if (gpu == 1){
			cudaStreamSynchronize(commandQueueGPU[nl[n]]);
		}
		for (i = i1; i < i2; i++){
			E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] = 0.;
			if (gpu == 1) for (z = z1; z < BS_3; z++){
				E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] += Buffersend1fine[nl[n]][(i - i1)*zsize + (z - z1)];
			}
			else for (z = z1; z < BS_3; z++) E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] += send1_fine[nl[n]][2 * (i - i1)*zsize + 2 * (z - z1) + 0];
			E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] /= (double)(BS_3);
		}
	}
	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		pack_send2_E(n, n, i1, i2, BS_2, BS_2 + D2, z1, z2, isize, zsize, send3_fine, E_corn, &(BufferE_1[nl[n]]), &(Buffersend3fine[nl[n]]), &(boundevent[nl[n]][398]));
		if (gpu == 1){
			cudaStreamSynchronize(commandQueueGPU[nl[n]]);
		}
		for (i = i1; i < i2; i++){
			E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] = 0.;
			if (gpu == 1) for (z = z1; z < BS_3; z++){
				E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] += Buffersend3fine[nl[n]][(i - i1)*zsize + (z - z1)];
			}
			else for (z = z1; z < BS_3; z++) E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] += send3_fine[nl[n]][2 * (i - i1)*zsize + 2 * (z - z1) + 0];
			E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i] /= (double)(BS_3);
		}
	}
}

void write_E_avg(double(*E_avg1)[BS_1 + 2 * N1G], double(*E_avg2)[BS_1 + 2 * N1G], int n){
	int i, i1, i2, z, z1, z2, isize, zsize, nz;
	//double  ph1, ph2;
	i1 = 0;
	i2 = BS_1 + N1G;
	z1 = 0;
	z2 = BS_3 + N3G;
	isize = (BS_1 + N1G);
	zsize = (BS_3 + N3G);
	nz = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]);

	#if (N_GPU>1)
	cudaSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		for (i = i1; i < i2; i++){
			if (gpu == 1)for (z = z1; z < z2; z++){
				Bufferrec1fine[nl[n]][(i - i1)*zsize + (z - z1)] = E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i];
			}
			else for (z = z1; z < z2; z++){
				receive1_fine[nl[n]][2 * (i - i1)*zsize + 2 * (z - z1) + 0] = E_avg1[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i];
			}
		}
		unpack_receive2_E(n, n, n, i1, i2, 0, D2, z1, z2, isize, zsize, receive1_fine, NULL, NULL, E_corn, &(BufferE_1[nl[n]]), &(Bufferrec1fine[nl[n]]), &(NULL_POINTER[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, 4, 0, 0, 0, 0);
	}
	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		for (i = i1; i < i2; i++){
			if (gpu == 1)for (z = z1; z < z2; z++){
				Bufferrec3fine[nl[n]][(i - i1)*zsize + (z - z1)] = E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i];
			}
			else for (z = z1; z < z2; z++){
				receive3_fine[nl[n]][2 * (i - i1)*zsize + 2 * (z - z1) + 0] = E_avg2[block[n][AMR_COORD1] * nz + block[n][AMR_COORD3]][i];
			}
		}
		unpack_receive2_E(n, n, n, i1, i2, BS_2, BS_2 + D2, z1, z2, isize, zsize, receive3_fine, NULL, NULL, E_corn, &(BufferE_1[nl[n]]), &(Bufferrec3fine[nl[n]]), &(NULL_POINTER[nl[n]]), &(NULL_POINTER[nl[n]]), NULL, 4, 0, 0, 0, 0);
	}
}

#endif

void const_transport2(double(*restrict psi[NB_LOCAL])[NDIM], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n){
	int i, j, z, k, ind0;
	#pragma omp parallel shared(n,n_ord,n_active,E_corn, gdet,psi,psf, dx,Dt, p, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0)
	{
		//update the staggered field components
		#pragma omp for collapse(3) schedule(static,(BS_1+D1)*(BS_2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][1] = psi[nl[n]][index_3D(n, i, j, z)][1] - Dt / dx[nl[n]][2] * (E_corn[nl[n]][index_3D(n, i, j + D2, z)][3] - E_corn[nl[n]][ind0][3]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
			#if(N3G>0)
			psf[nl[n]][index_3D(n, i, j, z)][1] += Dt / dx[nl[n]][3] * (E_corn[nl[n]][index_3D(n, i, j, z + D3)][2] - E_corn[nl[n]][ind0][2]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
			#endif
		}

		//update the staggered field components
		#pragma omp for collapse(3) schedule(static,(BS_1)*(BS_2+D2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][2] = psi[nl[n]][index_3D(n, i, j, z)][2] + Dt / dx[nl[n]][1] * (E_corn[nl[n]][index_3D(n, i + D1, j, z)][3] - E_corn[nl[n]][ind0][3]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
			#if(N3G>0)
			psf[nl[n]][index_3D(n, i, j, z)][2] += -Dt / dx[nl[n]][3] * (E_corn[nl[n]][index_3D(n, i, j, z + D3)][1] - E_corn[nl[n]][ind0][1]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
			#endif		
		}

		//update the staggered field components
		#if(N3G>0)
		#pragma omp for collapse(3) schedule(static,(BS_1)*(BS_2)*(BS_3+D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][3] = psi[nl[n]][index_3D(n, i, j, z)][3] - Dt / dx[nl[n]][1] * (E_corn[nl[n]][index_3D(n, i + D1, j, z)][2] - E_corn[nl[n]][ind0][2]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE3]
				+ Dt / dx[nl[n]][2] * (E_corn[nl[n]][index_3D(n, i, j + D2, z)][1] - E_corn[nl[n]][ind0][1]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE3];
		}
		#endif
	}
}

/***********************************************************************************************/
/***********************************************************************************************
flux_ct():
---------
-- performs the flux-averaging used to preserve the del.B = 0 constraint (see Toth 2000);
Note that we use in this new version of HARM dq instead of emf as temporary storage!

***********************************************************************************************/
void flux_ct(double(*restrict F1[NB_LOCAL])[NPR], double(*restrict F2[NB_LOCAL])[NPR], double(*restrict F3[NB_LOCAL])[NPR], int n)
{
	int i, j, z;
	int ind0;

	/* calculate EMFs */
	/* Toth approach: just average */
	#pragma omp parallel shared(n,dq, F1, F2, F3) private(i,j,z, ind0)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+D1)*(BS_2+D2)*(BS_3+D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1 + D1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + D2, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + D3){
			ind0 = index_3D(n, i, j, z);
			#if (N2G>0 && N3G>0)
			dq[nl[n]][ind0][1] = 0.25*(F2[nl[n]][ind0][B3] + F2[nl[n]][index_3D(n, i, j, z - 1)][B3] - F3[nl[n]][ind0][B2] - F3[nl[n]][index_3D(n, i, j - 1, z)][B2]);
			#endif
			#if (N1G>0 && N3G>0)
			dq[nl[n]][ind0][2] = 0.25*(F3[nl[n]][ind0][B1] + F3[nl[n]][index_3D(n, i - 1, j, z)][B1] - F1[nl[n]][ind0][B3] - F1[nl[n]][index_3D(n, i, j, z - 1)][B3]);
			#endif
			#if (N1G>0 && N2G>0)
			dq[nl[n]][ind0][3] = 0.25*(F1[nl[n]][ind0][B2] + F1[nl[n]][index_3D(n, i, j - 1, z)][B2] - F2[nl[n]][ind0][B1] - F2[nl[n]][index_3D(n, i - 1, j, z)][B1]);
			#else
			dq[nl[n]][ind0][3] = 0.25*(F1[nl[n]][ind0][B2] + F1[nl[n]][index_3D(n, i, j - 1, z)][B2]);
			#endif
		}

		/* rewrite EMFs as fluxes, after Toth */
		#pragma omp for collapse(3) schedule(static,(BS_1+D1)*(BS_2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1 + D1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) 	{
			ind0 = index_3D(n, i, j, z);
			#if (N1G>0)
			F1[nl[n]][ind0][B1] = 0.;
			#endif
			#if (N1G>0 && N2G>0)
			F1[nl[n]][ind0][B2] = 0.5*(dq[nl[n]][ind0][3] + dq[nl[n]][index_3D(n, i, j + 1, z)][3]);
			#endif
			#if (N1G>0 && N3G>0)
			F1[nl[n]][ind0][B3] = -0.5*(dq[nl[n]][ind0][2] + dq[nl[n]][index_3D(n, i, j, z + 1)][2]);
			#endif
		}

		#pragma omp for collapse(3) schedule(static,(BS_1)*(BS_2+D2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + D2, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) 	{
			ind0 = index_3D(n, i, j, z);
			#if (N1G>0 && N2G>0)		
			F2[nl[n]][ind0][B1] = -0.5*(dq[nl[n]][ind0][3] + dq[nl[n]][index_3D(n, i + 1, j, z)][3]);
			#endif
			#if (N2G>0 && N3G>0)
			F2[nl[n]][ind0][B3] = 0.5*(dq[nl[n]][ind0][1] + dq[nl[n]][index_3D(n, i, j, z + 1)][1]);
			#endif
			#if(N2G>0)
			F2[nl[n]][ind0][B2] = 0.;
			#endif
		}

		#pragma omp for collapse(3) schedule(static,(BS_1+D1)*(BS_2)*(BS_3+D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + D3) 	{
			ind0 = index_3D(n, i, j, z);
			#if (N1G>0 && N3G>0)
			F3[nl[n]][ind0][B1] = 0.5*(dq[nl[n]][ind0][2] + dq[nl[n]][index_3D(n, i + 1, j, z)][2]);
			#endif
			#if (N2G>0 && N3G>0)
			F3[nl[n]][ind0][B2] = -0.5*(dq[nl[n]][ind0][1] + dq[nl[n]][index_3D(n, i, j + 1, z)][1]);
			#endif
			#if(N3G>0)
			F3[nl[n]][ind0][B3] = 0.;
			#endif
		}
	}
}