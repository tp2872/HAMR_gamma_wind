#include "decs_MPI.h"

int AMR_coord_linear_RM(int level, int i, int j, int z);
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z);
void rm_order2(void);

void test_AMR(void){
}

int AMR_coord_linear(int level, int i, int j, int z){
	int index[N_LEVELS], coord[NDIM], factor[N_LEVELS], u, y, n;
	
	if (i < 0 || j < 0 || z < 0){
		n = -1;
		return n;
	}
	
	for (y = 0; y < N_LEVELS; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 1; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2 + REF_3) + 1;
		}
	}
	#if(REVERSE_ORDERING)
	index[0] = ((z - z % (int)pow(1 + REF_3, level)) / pow(1 + REF_3, level) * NB_2*NB_1 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level) * NB_1
		+ (i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level));
	n = index[0] * factor[0];
	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		coord[3] = (z % (int)(pow(1 + REF_3, level - u + 1)) - (z % (int)pow(1 + REF_3, level - u))) / pow(2, level - u);
		index[u] = coord[3] * (1 + REF_2)*(1 + REF_1) + coord[2] * (1 + REF_1) + coord[1]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u] + 1;
	}
	#else
	index[0] = ((i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level) * NB_2*NB_3 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level) * NB_3
		+ (z - z % (int)pow(1 + REF_3, level)) / pow(1 + REF_3, level));
	n = index[0] * factor[0];
	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		coord[3] = (z % (int)(pow(1 + REF_3, level - u + 1)) - (z % (int)pow(1 + REF_3, level - u))) / pow(2, level - u);
		index[u] = coord[1] * (1 + REF_2)*(1 + REF_3) + coord[2] * (1 + REF_3) + coord[3]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u]+1;
	}
	#endif

	return n;
}

int AMR_coord_linear_RM(int level, int i, int j, int z){
	int index[N_LEVELS], coord[NDIM], factor[N_LEVELS], u, y, n;

	if (i < 0 || j < 0 || z < 0){
		n = -1;
		return n;
	}

	for (y = 0; y < N_LEVELS; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 1; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2 + REF_3) + 1;
		}
	}

	index[0] = ((z - z % (int)pow(1 + REF_3, level)) / pow(1 + REF_3, level) * NB_2*NB_1 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level) * NB_1
		+ (i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level));
	n = index[0] * factor[0];
	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		coord[3] = (z % (int)(pow(1 + REF_3, level - u + 1)) - (z % (int)pow(1 + REF_3, level - u))) / pow(2, level - u);
		index[u] = coord[3] * (1 + REF_2)*(1 + REF_1) + coord[2] * (1 + REF_1) + coord[1]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u] + 1;
	}
	return n;
}


//Given a certain linear coordinate n this function determines the cartesian coordinates of a block and it's corresponding AMR-level
void AMR_coord_cart(int n, int *level, int *i, int *j, int *z){
	int ci[NDIM], cj[NDIM], cz[NDIM], factor[NDIM], index[N_LEVELS], number[N_LEVELS], y, u;

	for (y = 0; y < N_LEVELS; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 1; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2 + REF_3) + 1;
		}
	}

	for (y = 0; y < N_LEVELS; y++){
		number[y] = n-y;
		for (u = 0; u < y; u++){
			number[y] = number[y]%factor[u];
		}
		*level = y;
		if (number[y]%factor[y] == 0){
			break;
		}
	}
	for (y = 0; y <= (*level); y++){
		number[y] = n-(*level);
		for (u = 0; u < y; u++){
			number[y] = number[y] % factor[u];
		}
	}
	//fprintf(stderr, "Factor: %d \n", factor[0]);
	*i = 0;
	*j = 0;
	*z = 0;
	for (y = 0; y <= (*level); y++){
		#if(REVERSE_ORDERING)
		index[y] = (number[y] - number[y] % factor[y]) / factor[y];
		if (y == 0){
			ci[y] = (index[y] % (NB_2*NB_1) % NB_1);
			cj[y] = ((index[y] - ci[y]) % (NB_2*NB_1) / NB_1);
			cz[y] = (index[y] - (cj[y] * NB_1 + ci[y])) / (NB_2*NB_1);
		}
		else{
			ci[y] = (index[y] % ((REF_2 + 1)*(REF_1 + 1)) % (REF_1 + 1));
			cj[y] = ((index[y] - ci[y]) % ((REF_2 + 1)*(REF_1 + 1)) / (REF_1 + 1));
			cz[y] = (index[y] - (cj[y] * (REF_1 + 1) + ci[y])) / ((REF_2 + 1)*(REF_1 + 1));	
		}
		#else
		index[y] = (number[y] - number[y] % factor[y]) / factor[y];
		if (y == 0){
			cz[y] = (index[y] % (NB_2*NB_3) % NB_3);
			cj[y] = ((index[y] - cz[y]) % (NB_2*NB_3) / NB_3);
			ci[y] = (index[y] - (cj[y] * NB_3 + cz[y])) / (NB_2*NB_3);
		}
		else{
			cz[y] = (index[y] % ((REF_2 + 1)*(REF_3 + 1)) % (REF_3 + 1));
			cj[y] = ((index[y] - cz[y]) % ((REF_2 + 1)*(REF_3 + 1)) / (REF_3 + 1));
			ci[y] = (index[y] - (cj[y] * (REF_3 + 1) + cz[y])) / ((REF_2 + 1)*(REF_3 + 1));
		}
		#endif

		*i += ci[y] * pow(1 + REF_1, (*level - y));
		*j += cj[y] * pow(1 + REF_2, (*level - y));
		*z += cz[y] * pow(1 + REF_3, (*level - y));
	}
}

//Given a certain linear coordinate n this function determines the cartesian coordinates of a block and it's corresponding AMR-level
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z){
	int ci[NDIM], cj[NDIM], cz[NDIM], factor[NDIM], index[N_LEVELS], number[N_LEVELS], y, u;

	for (y = 0; y < N_LEVELS; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 1; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2 + REF_3) + 1;
		}
	}

	for (y = 0; y < N_LEVELS; y++){
		number[y] = n - y;
		for (u = 0; u < y; u++){
			number[y] = number[y] % factor[u];
		}
		*level = y;
		if (number[y] % factor[y] == 0){
			break;
		}
	}
	for (y = 0; y <= (*level); y++){
		number[y] = n - (*level);
		for (u = 0; u < y; u++){
			number[y] = number[y] % factor[u];
		}
	}
	//fprintf(stderr, "Factor: %d \n", factor[0]);
	*i = 0;
	*j = 0;
	*z = 0;
	for (y = 0; y <= (*level); y++){
		index[y] = (number[y] - number[y] % factor[y]) / factor[y];
		if (y == 0){
			ci[y] = (index[y] % (NB_2*NB_1) % NB_1);
			cj[y] = ((index[y] - ci[y]) % (NB_2*NB_1) / NB_1);
			cz[y] = (index[y] - (cj[y] * NB_1 + ci[y])) / (NB_2*NB_1);
		}
		else{
			ci[y] = (index[y] % ((REF_2 + 1)*(REF_1 + 1)) % (REF_1 + 1));
			cj[y] = ((index[y] - ci[y]) % ((REF_2 + 1)*(REF_1 + 1)) / (REF_1 + 1));
			cz[y] = (index[y] - (cj[y] * (REF_1 + 1) + ci[y])) / ((REF_2 + 1)*(REF_1 + 1));
		}
		*i += ci[y] * pow(1 + REF_1, (*level - y));
		*j += cj[y] * pow(1 + REF_2, (*level - y));
		*z += cz[y] * pow(1 + REF_3, (*level - y));
	}
}

//Sets the AMR hierarchy
void set_AMR(void){
	int n, n_parent, n_child[9], n_nbr[21], level, i, j, z, i1, j1, z1,
		i_max, j_max, z_max, i_parent, j_parent, z_parent, ind;
	int y, rem, node = 0;
	block = (int(*)[NV])calloc(NB, sizeof(int[NV]));
	max_levels = 0;

	//find maximum block number
	n_max=AMR_coord_linear(N_LEVELS - 1, pow(REF_1 + 1, N_LEVELS - 1)*NB_1 - 1, pow(REF_2 + 1, N_LEVELS - 1)*NB_2 - 1, pow(REF_3 + 1, N_LEVELS - 1)*NB_3 - 1);

	for (i = 0; i < NB_LOCAL; i++) mem_spot[i] = -1;

 	//Set all 'one-time'parameters of all blocks (refined and unrefined)
	for (n = 0; n <= n_max; n++){

		//Find level and coordinates of the respective block
		AMR_coord_cart(n, &level, &i, &j, &z);
		block[n][AMR_LEVEL] = level;
		block[n][AMR_COORD1] = i;
		block[n][AMR_COORD2] = j;
		block[n][AMR_COORD3] = z;

		//Set maximum coordinates
		i_max = NB_1*pow(1 + REF_1, level) - 1;
		j_max = NB_2*pow(1 + REF_2, level) - 1;
		z_max = NB_3*pow(1 + REF_3, level) - 1;

		//Find parent of block
		if (block[n][AMR_LEVEL] == 0) block[n][AMR_PARENT] = -1; //-1 means no parent
		else{
			i_parent = (i - i % (1 + REF_1)) / (1 + REF_1);
			j_parent = (j - j % (1 + REF_2)) / (1 + REF_2);
			z_parent = (z - z % (1 + REF_3)) / (1 + REF_3);
			block[n][AMR_PARENT] = AMR_coord_linear(level - 1, i_parent, j_parent, z_parent);
		}

		//Find children of block, -1 means no children
		if (block[n][AMR_LEVEL] == N_LEVELS - 1){
			block[n][AMR_CHILD1] = -1;
			block[n][AMR_CHILD2] = -1;
			block[n][AMR_CHILD3] = -1;
			block[n][AMR_CHILD4] = -1;
			block[n][AMR_CHILD5] = -1;
			block[n][AMR_CHILD6] = -1;
			block[n][AMR_CHILD7] = -1;
			block[n][AMR_CHILD8] = -1;
		}
		else{
			block[n][AMR_CHILD1] = AMR_coord_linear(level + 1, i * (1 + REF_1), j * (1 + REF_2), z * (1 + REF_3));

			block[n][AMR_CHILD2] = AMR_coord_linear(level + 1, i * (1 + REF_1), j * (1 + REF_2), z * (1 + REF_3) + REF_3);

			block[n][AMR_CHILD3] = AMR_coord_linear(level + 1, i * (1 + REF_1), j * (1 + REF_2) + REF_2, z * (1 + REF_3));

			block[n][AMR_CHILD4] = AMR_coord_linear(level + 1, i * (1 + REF_1), j * (1 + REF_2) + REF_2, z * (1 + REF_3) + REF_3);

			block[n][AMR_CHILD5] = AMR_coord_linear(level + 1, i * (1 + REF_1) + REF_1, j * (1 + REF_2), z * (1 + REF_3));

			block[n][AMR_CHILD6] = AMR_coord_linear(level + 1, i * (1 + REF_1) + REF_1, j * (1 + REF_2), z * (1 + REF_3) + REF_3);

			block[n][AMR_CHILD7] = AMR_coord_linear(level + 1, i * (1 + REF_1) + REF_1, j * (1 + REF_2) + REF_2, z * (1 + REF_3));

			block[n][AMR_CHILD8] = AMR_coord_linear(level + 1, i * (1 + REF_1) + REF_1, j * (1 + REF_2) + REF_2, z * (1 + REF_3) + REF_3);
		}

		//Find neighbours of block
		if (j - 1 < 0 && PERIODIC2 == 1) j1 = j_max;
		else j1 = j - 1;
		block[n][AMR_NBR1] = AMR_coord_linear(level, i, j1, z);

		if (i + 1 > i_max && PERIODIC1 == 1) i1 = 0;
		else if (i + 1 > i_max && PERIODIC1 == 0) i1 = -1;
		else i1 = i + 1;
		block[n][AMR_NBR2] = AMR_coord_linear(level, i1, j, z);

		if (j + 1 > j_max && PERIODIC2 == 1) j1 = 0;
		else if (j + 1 > j_max && PERIODIC2 == 0) j1 = -1;
		else j1 = j + 1;
		block[n][AMR_NBR3] = AMR_coord_linear(level, i, j1, z);

		if (i - 1 < 0 && PERIODIC1 == 1) i1 = i_max;
		else i1 = i - 1;
		block[n][AMR_NBR4] = AMR_coord_linear(level, i1, j, z);

		if (z + 1 > z_max && PERIODIC3 == 1) z1 = 0;
		else if (z + 1 > z_max && PERIODIC3 == 0 ) z1 = -1;
		else z1 = z + 1;
		block[n][AMR_NBR5] = AMR_coord_linear(level, i, j, z1);

		if (z - 1 < 0 && PERIODIC3 == 1 ) z1 = z_max;
		else z1 = z - 1;
		block[n][AMR_NBR6] = AMR_coord_linear(level, i, j, z1);
		
		block[n][AMR_POLE] = 0; //If there are no transmissive boundary conditions no special treatment of the pole is necessary
		
		//Find the neighbours in the case we have transmissive boundary conditions at the pole
		#if (TRANS_BOUND)
		if (NB_3 % 2 != 0) fprintf(stderr, "Number of blocks in the third dimension is not an even number. This is incompatible with TRANS_BOUND");

		//First tell the code if you are dealing with a pole at theta=0 (1) or at theta=Pi (2)
		if (j == 0){
			block[n][AMR_POLE] += 1;
			block[n][AMR_NBR1] = AMR_coord_linear(level, i, j, (z + NB_3*(int)pow(1 + REF_3, level) / 2) % (z_max + 1));
		}
		if (j == j_max){
			block[n][AMR_POLE] += 2;
			block[n][AMR_NBR3] = AMR_coord_linear(level, i, j, (z + NB_3*(int)pow(1 + REF_3, level) / 2) % (z_max + 1));
		}
		#endif

		//Find corners of block assuming only third dimension is periodic
		//x-y plane
		if (i + 1 > i_max || j - 1 < 0){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i + 1;
			j1 = j - 1;
		}
		block[n][AMR_CORN1] = AMR_coord_linear(level, i1, j1, z);

		if (i + 1 > i_max || j + 1 > j_max){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i + 1;
			j1 = j + 1;
		}
		block[n][AMR_CORN2] = AMR_coord_linear(level, i1, j1, z);
		
		if (i - 1 < 0|| j + 1 > j_max){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i - 1;
			j1 = j + 1;
		}
		block[n][AMR_CORN3] = AMR_coord_linear(level, i1, j1, z);

		if (i - 1 < 0 || j - 1 < 0){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i - 1;
			j1 = j - 1;
		}
		block[n][AMR_CORN4] = AMR_coord_linear(level, i1, j1, z);
		
		//x-z plane
		i1 = i + 1;
		z1 = z - 1;
		if (i + 1 > i_max) i1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN5] = AMR_coord_linear(level, i1, j, z1);

		i1 = i + 1;
		z1 = z + 1;
		if (i + 1 > i_max) i1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN6] = AMR_coord_linear(level, i1, j, z1);
		
		i1 = i - 1;
		z1 = z + 1;
		if (i - 1 < 0) i1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN7] = AMR_coord_linear(level, i1, j, z1);


		i1 = i - 1;
		z1 = z - 1;
		if (i - 1 < 0) i1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN8] = AMR_coord_linear(level, i1, j, z1);
		
		//y-z plane
		j1 = j - 1;
		z1 = z + 1;
		if (j - 1 < 0) j1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN9] = AMR_coord_linear(level, i, j1, z1);

		j1 = j + 1;
		z1 = z + 1;
		if (j + 1 > j_max) j1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN10] = AMR_coord_linear(level, i, j1, z1);

		
		j1 = j + 1;
		z1 = z - 1;
		if (j + 1 > j_max) j1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN11] = AMR_coord_linear(level, i, j1, z1);


		j1 = j - 1;
		z1 = z - 1;
		if (j - 1 < 0) j1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN12] = AMR_coord_linear(level, i, j1, z1);
	}

	//Set offsets and size of blocks
	for (n = 0; n <= n_max; n++){
		N1_GPU_offset[n] = block[n][AMR_COORD1] * BS_1;
		N2_GPU_offset[n] = block[n][AMR_COORD2] * BS_2;
		N3_GPU_offset[n] = block[n][AMR_COORD3] * BS_3;
	}

	for (n = 0; n <= n_max; n++){
		set_points(n);

		//For the moment don't refine any block
		block[n][AMR_REFINED] = 0;

		block[n][GDUMP_WRITTEN] = 0;

		//No node assigned yet
		block[n][AMR_NODE] = -1;

		//No special GPU assigned yet
		block[n][AMR_GPU] = local_rank%numdevices;

		//For the moment only activate the 0 level blocks
		if (block[n][AMR_LEVEL] == 0){
			block[n][AMR_ACTIVE] = 1;
			block[n][AMR_TIMELEVEL] = 1;
		}
		else{
			block[n][AMR_ACTIVE] = 0;
			block[n][AMR_TIMELEVEL] = 1;
		}
	}

	//Check if there is a restart file with the preset grid hierarchy
	restart_read_param();
	#if(READ_OLD)
	restart_read_grid();
	#endif
	activate_blocks();
	set_corners();

	MPI_Barrier(MPI_COMM_WORLD);

	balance_load();

}

void balance_load(void){
	int i,j, z, node, tt, fp, ip, y, rem, nr_timesteps, n_active_localsteps[NB], max_timelevel;
	int i1, j1, z1, k, n, u, b, g, stride;
	int n_active_total_steps = 0, n_active_total_steps_t[10];
	int steps_total_RM[NB];
	int NODE[NB];
	int n_active_total_t[10];
	int (*n_ord_total_RM_t)[10];
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	int timelevel_cutoff = AMR_MAXTIMELEVEL;
	#if(DEREFINE_POLE)
	rm_order();
	#else
	rm_order2();
	#endif
	n_ord_total_RM_t=(int(*)[10])calloc(NB, sizeof(int[10]));
	int n_active_local_max = 0;
	int n_active_local_min = 1;

	do{
		//Strictly adhere to z-order
		/*if ((n_active_local_max / ((double)AMR_MAXTIMELEVEL) > MAX_BLOCKS || (n_active_local_min == 0 && n_active_total > numtasks)) && timelevel_cutoff >= 2)timelevel_cutoff /= 2;
		if (n_active_local_max > MAX_BLOCKS || (n_active_local_min == 0 && n_active_total > numtasks)) timelevel_cutoff /= 2;
		n_active_total_steps = 0;
		n_active_local_max = 0;
		n_active_local_min = 0;
		for (n = 0; n < n_active_total; n++){
		steps_total_RM[n] = n_active_total_steps + AMR_MAXTIMELEVEL / 2 / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff);
		n_active_total_steps += AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff);
		}

		rem = n_active_total_steps % (numtasks); //remainder of last unfilled block
		y = (n_active_total_steps - rem) / (numtasks); //number of blocks/node
		tt = -1, ip = 0, fp = 0;

		//First use non blocking sends and receives to send and receive data around cluster
		for (i = 0; i < n_active_total; i++){
		NODE[n_ord_total_RM[i]] = (steps_total_RM[i] - steps_total_RM[i] % (y + 1)) / (y + 1);
		if (NODE[n_ord_total_RM[i]] >= rem){
		if (tt == -1){
		fp = NODE[n_ord_total[i]];
		ip = steps_total_RM[i] - steps_total_RM[i] % (y + 1);
		tt = 0;
		}
		NODE[n_ord_total_RM[i]] = fp + ((steps_total_RM[i] - ip) - (steps_total_RM[i] - ip) % y) / y;
		}
		if (NODE[n_ord_total_RM[i]] >= numtasks) fprintf(stderr, "Error balance_load() \n");
		if (NODE[n_ord_total_RM[i]] == rank){
		n_active_local_max++;
		n_active_local_min = n_active_local_max;
		}
		}
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);*/

		//Load balance for uniform load over timesteps
		/*if (n_active_local_max > MAX_BLOCKS / ((double)AMR_MAXTIMELEVEL) || (n_active_local_min == 0 && n_active_total > numtasks)) timelevel_cutoff /= 2;
		n_active_total_steps = 0;
		n_active_local_max = 0;
		n_active_local_min = 0;
		max_timelevel = 0;
		for (n = 0; n < n_active_total; n++){
		n_active_total_steps += AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff);
		NODE[n_ord_total_RM[n]] = -1;
		max_timelevel = MY_MAX(max_timelevel, MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff));
		}
		for (u = 0; u < numtasks; u++) n_active_localsteps[u]=0;

		u = 0;
		for (i = 0; i < n_active_total; i++){ //Allocate a node to the lower timelevels
		if (block[n_ord_total_RM[i]][AMR_TIMELEVEL] != max_timelevel){
		nr_timesteps = AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[i]][AMR_TIMELEVEL], timelevel_cutoff);
		if (n_active_localsteps[u] + nr_timesteps <= n_active_total_steps / numtasks){
		n_active_localsteps[u] += nr_timesteps;
		NODE[n_ord_total_RM[i]] = u;
		}
		else if (u<numtasks - 1){
		while (u<numtasks - 1){
		u++;
		if (n_active_localsteps[u] + nr_timesteps <= n_active_total_steps / numtasks){
		n_active_localsteps[u] += nr_timesteps;
		NODE[n_ord_total_RM[i]] = u;
		break;
		}
		}
		}
		}
		}
		u = 0;

		for (b = 0; b <= log(AMR_MAXTIMELEVEL)/log(2); b++){
		for (i = 0; i < n_active_total; i++){ //Allocate a node to the higher timelevels as long node is not full
		if (block[n_ord_total_RM[i]][AMR_TIMELEVEL] == pow(2,b) && NODE[n_ord_total_RM[i]]==-1){
		nr_timesteps = AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[i]][AMR_TIMELEVEL], timelevel_cutoff);
		if (n_active_localsteps[u] + nr_timesteps <= n_active_total_steps / numtasks){
		n_active_localsteps[u] += nr_timesteps;
		NODE[n_ord_total_RM[i]] = u;
		}
		else if (u < numtasks - 1){
		while (u < numtasks - 1){
		u++;
		if (n_active_localsteps[u] + nr_timesteps <= n_active_total_steps / numtasks){
		n_active_localsteps[u] += nr_timesteps;
		NODE[n_ord_total_RM[i]] = u;
		break;
		}
		}
		}
		}
		}
		}
		u = 0;
		for (i = 0; i < n_active_total; i++){ //Allocate a node to the remaining tiles
		if (NODE[n_ord_total_RM[i]] == -1){
		nr_timesteps = AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[i]][AMR_TIMELEVEL], timelevel_cutoff);
		n_active_localsteps[u] += nr_timesteps;
		NODE[n_ord_total_RM[i]] = u%numtasks;
		u++;
		}
		}

		n_active_local_max = n_active_localsteps[rank];
		n_active_local_min = n_active_local_max;
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);*/

		/*First make a z-order curve for each timelevel seperately, then load balance for timesteps. This is the best and most advanced method*/
		if ((n_active_local_max / ((double)AMR_MAXTIMELEVEL) > MAX_BLOCKS || (n_active_local_min == 0 && n_active_total > numtasks)) && timelevel_cutoff >= 2)timelevel_cutoff /= 2;
		for (i = 0; i <= round(log(AMR_MAXTIMELEVEL) / log(2)); i++){
			n_active_total_t[i] = 0;
			n_active_total_steps_t[i] = 0;
		}
		int tl;
		//Order active blocks in an ordered array and keep track of the number osf blocks and timesteps at each timelevel
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total_RM[n]][AMR_ACTIVE] == 1){
				tl = MY_MIN(round(log(block[n_ord_total_RM[n]][AMR_TIMELEVEL]) / log(2)), log(timelevel_cutoff) / log(2));
				n_ord_total_RM_t[n_active_total_t[tl]][tl] = n_ord_total_RM[n];
				n_active_total_steps_t[tl] += AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff);
				n_active_total_t[tl]++;
			}
		}

		//Reset variables
		n_active_local_max = 0;
		n_active_local_min = 0;
		for (u = 0; u < numtasks; u++) n_active_localsteps[u] = 0;

		int increment = 0, j, n0, fillup_mode = 0;
		u = 0; //Initial node number
		int sw = 0;

		//Try to give each node the same number of lower timelevel blocks
		for (i = 0; i <= round(log(timelevel_cutoff) / log(2)); i++){

			//If there is not an even load from the previous timelevel, first correct for that
			if (n_active_localsteps[(u - 1 + numtasks) % numtasks] > n_active_localsteps[u % numtasks]){
				nr_timesteps = AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM_t[0][i]][AMR_TIMELEVEL], timelevel_cutoff);
				increment = (n_active_localsteps[(u - 1 + numtasks) % numtasks] - n_active_localsteps[u % numtasks]) / nr_timesteps;
				fillup_mode = 1;
			}
			if (n_active_localsteps[(u - 1 + numtasks) % numtasks] == n_active_localsteps[u % numtasks]){ //Otherwise just do nornmal load balancing
				rem = n_active_total_t[i] % (numtasks); //remainder number of blocks at given timelevel
				increment = (n_active_total_t[i] - rem) / (numtasks); //number of blocks/node at given timelevel
				fillup_mode = 0;
				sw = 1;
			}
			n = 0;
			while (n < n_active_total_t[i]){

				nr_timesteps = AMR_MAXTIMELEVEL / MY_MIN(block[n_ord_total_RM_t[n][i]][AMR_TIMELEVEL], timelevel_cutoff);
				if (fillup_mode == 1) increment = (n_active_localsteps[(u - 1 + numtasks) % numtasks] - n_active_localsteps[u % numtasks]) / nr_timesteps;
				if (n_active_localsteps[(u - 1 + numtasks) % numtasks] == n_active_localsteps[u % numtasks]){
					rem = (n_active_total_t[i] - n) % (numtasks); //remainder number of blocks at given timelevel
					increment = (n_active_total_t[i] - n - rem) / (numtasks); //number of blocks/node at given timelevel
					fillup_mode = 0;
					sw = 1;
				}

				if (fillup_mode == 0 && ((n_active_total_t[i] - n) / (increment + 1)) == rem && (n_active_total_t[i] - n) % (increment + 1) == 0 && rem>0){
					increment += 1;
					sw = 1;
				}
				increment = MY_MIN(increment, n_active_total_t[i] - n);
				for (j = 0; j < increment; j++){
					NODE[n_ord_total_RM_t[n + j][i]] = u%numtasks;
					n_active_localsteps[u%numtasks] += nr_timesteps;
					if (rank == u%numtasks)n_active_local_max++;
				}
				n += increment;
				if (n_active_localsteps[u%numtasks] == n_active_localsteps[(u - 1 + numtasks) % numtasks] || sw == 1){
					sw = 0;
					u = (u + 1) % numtasks;//Increase node number
				}
				if (increment == 0) printf("error");
			}
		}
		n_active_local_min = n_active_local_max;
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	} while ((n_active_local_max > MAX_BLOCKS || (n_active_local_min == 0 && n_active_total > numtasks)) && timelevel_cutoff >= 2);
	if (rank == 0 && timelevel_cutoff != AMR_MAXTIMELEVEL) fprintf(stderr, "Error in balance_load. Due to too little/many blocks the maximum timelevel can't be honoured and the hierarchical timestepping is downgraded! \n");
	if (rank == 0 && (n_active_local_max > MAX_BLOCKS)) fprintf(stderr, "Error in balance_load: Too many blocks refined, possible to get OpenCL or OOM errors! \n");
	if (rank == 0) fprintf(stderr, "Load balance started, timelevel_cutoff %d! \n", timelevel_cutoff);
	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[n_ord_total_RM[i]]){
			if (block[n_ord_total_RM[i]][AMR_NODE] == rank){
				rc = MPI_Isend(&p[nl[n_ord_total_RM[i]]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[i]], (2 * n_ord_total_RM[i] + 0) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[i]]][0]);
				#if STAGGERED
				rc += MPI_Isend(&ps[nl[n_ord_total_RM[i]]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[i]], (2 * n_ord_total_RM[i] + 1) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[i]]][1]);
				#endif
				if (rc != 0)fprintf(stderr, "Error balance_load send %d", rc);
			}
		}
	}

	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[n_ord_total_RM[i]]){
			if (NODE[n_ord_total_RM[i]] == rank){
				//Allocate memory for active blocks on node
				temp_p[n_ord_total_RM[i]] = (double(*)[NPR])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NPR]));
				temp_ps[n_ord_total_RM[i]] = (double(*)[NDIM])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NDIM]));
				if (block[n_ord_total_RM[i]][AMR_NODE] >= 0){
					rc = MPI_Irecv(&temp_p[n_ord_total_RM[i]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (2 * n_ord_total_RM[i] + 0) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp1[n_ord_total_RM[i]]);
					#if STAGGERED
					rc += MPI_Irecv(&temp_ps[n_ord_total_RM[i]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (2 * n_ord_total_RM[i] + 1) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp2[n_ord_total_RM[i]]);
					#endif
					if (rc != 0)fprintf(stderr, "Error balance_load receive %d", rc);
				}
			}
		}
	}

	for (i = 0; i < n_active_total; i++){
		//Then use MPI_wait to clean up data that has been sent
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[n_ord_total_RM[i]]){
			if (block[n_ord_total_RM[i]][AMR_NODE] == rank){
				MPI_Wait(&boundreqs[nl[n_ord_total_RM[i]]][0], &Statbound[nl[n_ord_total_RM[i]]][0]);
				#if STAGGERED
				MPI_Wait(&boundreqs[nl[n_ord_total_RM[i]]][1], &Statbound[nl[n_ord_total_RM[i]]][1]);
				#endif
				#if(GPU_ENABLED || GPU_DEBUG )
				GPU_finish(n_ord_total_RM[i]);
				#endif
				free_arrays(n_ord_total_RM[i]);		
			}
		}
	}

	//Then use MPI_wait to receive data 
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total_RM[n]][AMR_NODE] != NODE[n_ord_total_RM[n]]){
			if (NODE[n_ord_total_RM[n]] == rank){
				if (block[n_ord_total_RM[n]][AMR_NODE] >= 0){
					MPI_Wait(&boundreqstemp1[n_ord_total_RM[n]], &Statbound[0][10]);
					#if STAGGERED
					MPI_Wait(&boundreqstemp2[n_ord_total_RM[n]], &Statbound[0][11]);
					#endif
				}
				set_arrays(n_ord_total_RM[n]);
				set_grid(n_ord_total_RM[n]);
				#pragma omp parallel for schedule(dynamic,1)  private(i, j, z, k)
				ZSLOOP3D(N1_GPU_offset[n_ord_total_RM[n]] - N1G, N1_GPU_offset[n_ord_total_RM[n]] + BS_1 - 1 + N1G, N2_GPU_offset[n_ord_total_RM[n]] - N2G, N2_GPU_offset[n_ord_total_RM[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord_total_RM[n]] - N3G, N3_GPU_offset[n_ord_total_RM[n]] + BS_3 - 1 + N3G){
					PLOOP p[nl[n_ord_total_RM[n]]][index_3D(n_ord_total_RM[n], i, j, z)][k] = temp_p[n_ord_total_RM[n]][index_3D(n_ord_total_RM[n], i, j, z)][k];
					for(k=0; k<NDIM; k++) ps[nl[n_ord_total_RM[n]]][index_3D(n_ord_total_RM[n], i, j, z)][k] = temp_ps[n_ord_total_RM[n]][index_3D(n_ord_total_RM[n], i, j, z)][k];
				}
				free(temp_p[n_ord_total_RM[n]]);
				free(temp_ps[n_ord_total_RM[n]]);
				#if(GPU_ENABLED || GPU_DEBUG )
				set_arrays_GPU(n_ord_total_RM[n], block[n_ord_total_RM[n]][AMR_GPU]);
				GPU_write(n_ord_total_RM[n]);
				#endif
			}
		}
	}

	for (i = 0; i < n_active_total; i++){
		block[n_ord_total_RM[i]][AMR_NODE] = NODE[n_ord_total_RM[i]];
	}

	activate_blocks();
	count_node[0]=0;
	for (n = 0; n < n_active; n++){
		count_node[0] += AMR_MAXTIMELEVEL / block[n_ord_RM[nl[n]]][AMR_TIMELEVEL];
	}

	int min1 = n_active;
	int max1 = n_active;
	int min_steps = count_node[0];
	int max_steps = count_node[0];
	int total_steps = count_node[0];
	MPI_Allreduce(MPI_IN_PLACE, &min1, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &max1, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &min_steps, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &max_steps, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &total_steps, 1, MPI_INT, MPI_SUM, mpi_cartcomm);

	if (rank == 0) fprintf(stderr, "Number of active blocks (total, min,max): %d %d %d \n", n_active_total, min1, max1);
	if (rank == 0) fprintf(stderr, "Number of active steps (total, min,max): %d %d %d \n", total_steps, min_steps, max_steps);

	bound_prim(p, 1);
	#if(GPU_ENABLED)
	GPU_boundprim(1);
	#endif
	free(n_ord_total_RM_t);
	if (rank == 0) fprintf(stderr, "Load balance finished! \n");
}

void balance_load_gpu(void){
	/*int i, g, tt, fp, ip, y, q, rem, n, gpu;
	int n_active_steps = 0;
	int steps_RM[NB];

	//Find the appropriate GPU assuming Z ordering
	#if(DEREFINE_POLE)
	rm_order();
	#else
	rm_order2();
	#endif

	//Loop over number of GPUs
	for (n = 0; n < n_active; n++){
		steps_RM[nl[n]] = n_active_steps + AMR_MAXTIMELEVEL / 2 / block[n_ord_RM[nl[n]]][AMR_TIMELEVEL];
		n_active_steps += AMR_MAXTIMELEVEL / block[n_ord_RM[nl[n]]][AMR_TIMELEVEL];
	}

	tt = -1, ip = 0, fp = 0, rem, gpu = 0;
	rem = n_active_steps % (NQ); //remainder of last unfilled block
	y = (n_active_steps - rem) / (NQ); //number of blocks/gpu

	for (n = 0; n < n_active; n++){
		gpu = (steps_RM[nl[n]] - steps_RM[nl[n]] % (y + 1)) / (y + 1);
		if (gpu >= rem){
			if (tt == -1){
				fp = gpu;
				ip = steps_RM[nl[n]] - steps_RM[nl[n]] % (y + 1);
				tt = 0;
			}
			gpu = fp + ((steps_RM[nl[n]] - ip) - (steps_RM[nl[n]] - ip) % y) / y;
		}
		if (gpu >= NQ) fprintf(stderr, "Error balance_load_gpu() \n");
		//commandQueueGPU[n_ord_RM[nl[n]]] = commandQueue[gpu];
	}*/
}

/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
void activate_blocks(void){
	int n, i;

	n_active = 0;
	n_active_total = 0;
	MPI_Barrier(MPI_COMM_WORLD);

	for (n = 0; n <= n_max; n++){
		if (block[n][AMR_ACTIVE] == 1 && block[n][AMR_NODE] == rank){
			//Order active blocks into array n_ord and keep track of number of active block in n_active
			n_ord[n_active] = n;
			n_ord_RM[n_active] = n;
			n_active++;
			
		}
		if (block[n][AMR_ACTIVE] == 1){
			//Order active blocks into array n_ord and keep track of number of active block in n_active_total
			n_ord_total[n_active_total] = n;
			n_ord_total_RM[n_active_total] = n;
			block[n][AMR_NUMBER] = n_active_total;
			n_active_total++;
			if (block[n][AMR_LEVEL] > 0) block[block[n][AMR_PARENT]][AMR_REFINED] = 1;
			block[n][AMR_REFINED] = 0;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

}

void block_average(int n, int n_child, int i1, int i2, int j1, int j2, int z1, int z2){
	int i, j, z, k, ic, jc, zc, i_1, i_2, j_1, j_2, z_1, z_2;
	struct of_geom geom;
	struct of_state q;

	#pragma omp parallel private(i, j, z, k, ic, jc, zc, i_1, i_2, j_1, j_2, z_1, z_2,q,geom)
	{
		//Average primitive quantities
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 0.125 * (
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k]);
					}
				}
			}
		}

		//Average conserved quantitites
		/*#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					for (k = 0; k < NPR; k++){
						i_1 = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child];
						i_2 = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1;
						j_1 = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child];
						j_2 = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2;
						z_1 = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child];
						z_2 = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3;

						get_geometry(n_child, i_1, j_1, z_1, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)]);

						get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]);
						
						get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)]);

						get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)]);
						
						get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)]);
						
						get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)]);
						
						get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)]);
						
						get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
						get_state(p[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)], &geom, &q);
						primtoU(p[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)]);

						dq[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 0.125  * (
							dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)][k] +
							dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)][k] +
							dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)][k] +
							dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)][k] +
							dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)][k] +
							dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)][k] +
							dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)][k] +
							dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)][k] );
					}
					pflag[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])] = Utoprim_2d(dq[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])], geom.gcov, geom.gcon, geom.g, p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])]);
				}
			}
		}*/
		#if STAGGERED
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2 + D1; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					ic = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child];
					k = 1;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + REF_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + REF_2, zc)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc + REF_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + REF_3)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + REF_2, zc + REF_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + REF_2, zc + REF_3)][FACE1]);
				}
			}
		}
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2 + D2; j++){
				for (z = z1; z < z2; z++){
					ic = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child];
					k = 2;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic + REF_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + REF_1, jc, zc)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc + REF_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + REF_3)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic + REF_1, jc, zc + REF_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic + REF_1, jc, zc + REF_3)][FACE2]);
				}
			}
		}
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2 + D3; z++){
					ic = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child];
					k = 3;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic + REF_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + REF_1, jc, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + REF_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + REF_2, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic + REF_1, jc + REF_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + REF_1, jc + REF_2, zc)][FACE3]);
				}
			}
		}
		#endif
	}
}

void derefine(int n){
	int i,j,z,k, n_child;
	if (rank == 0) fprintf(stderr, "Derefining block %d %d %d %d \n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
	if (block[n][AMR_ACTIVE] != 0) fprintf(stderr, "Error: Trying to derefine active block %d! \n", n);

	block[n][AMR_ACTIVE] = 1;
	if (block[n][AMR_NODE] == rank){
		set_arrays(n);
		set_grid(n);

		if (block[n][AMR_CHILD1] >= 0){
			n_child = block[n][AMR_CHILD1];
			block_average(n, n_child, 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD1]);
			GPU_finish(block[n][AMR_CHILD1]);
			#endif
		}

		if (block[n][AMR_CHILD2] >= 0 && REF_3 == 1){
			n_child = block[n][AMR_CHILD2];
			block_average(n, n_child, 0, BS_1 / (1 + REF_1), 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD2]);
			GPU_finish(block[n][AMR_CHILD2]);
			#endif
		}

		if (block[n][AMR_CHILD3] >= 0 && REF_2 == 1){
			n_child = block[n][AMR_CHILD3];
			block_average(n, n_child, 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD3]);
			GPU_finish(block[n][AMR_CHILD3]);
			#endif
		}

		if (block[n][AMR_CHILD4] >= 0 && REF_2 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD4];
			block_average(n, n_child, 0, BS_1 / (1 + REF_1), BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD4]);
			GPU_finish(block[n][AMR_CHILD4]);
			#endif
		}

		if (block[n][AMR_CHILD5] >= 0 && REF_1 == 1){
			n_child = block[n][AMR_CHILD5];
			block_average(n, n_child, BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), 0, BS_3 / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD5]);
			GPU_finish(block[n][AMR_CHILD5]);
			#endif
		}

		if (block[n][AMR_CHILD6] >= 0 && REF_1 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD6];
			block_average(n, n_child, BS_1 / (1 + REF_1), BS_1, 0, BS_2 / (1 + REF_2), BS_3 / (1 + REF_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD6]);
			GPU_finish(block[n][AMR_CHILD6]);
			#endif
		}

		if (block[n][AMR_CHILD7] >= 0 && REF_1 == 1 && REF_2 == 1){
			n_child = block[n][AMR_CHILD7];
			block_average(n, n_child, BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, 0, BS_3 / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD7]);
			GPU_finish(block[n][AMR_CHILD7]);
			#endif
		}

		if (block[n][AMR_CHILD8] >= 0 && REF_1 == 1 && REF_2 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD8];
			block_average(n, n_child, BS_1 / (1 + REF_1), BS_1, BS_2 / (1 + REF_2), BS_2, BS_3 / (1 + REF_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD8]);
			GPU_finish(block[n][AMR_CHILD8]);
			#endif
		}
	}
	int min_timelevel = AMR_MAXTIMELEVEL;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
		if (block[block[n][i]][AMR_TIMELEVEL] < min_timelevel) min_timelevel = block[block[n][i]][AMR_TIMELEVEL];
	}
	block[n][AMR_TIMELEVEL] = MY_MIN(2 * min_timelevel, AMR_MAXTIMELEVEL);
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_ACTIVE] = 0;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_TIMELEVEL] = 1;
	
	#if(GPU_ENABLED || GPU_DEBUG )
	if (block[n][AMR_NODE] == rank){
		set_arrays_GPU(n, block[n][AMR_GPU]);
		GPU_write(n);
	}
	#endif
}

void refine_cell(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict prim[NB_LOCAL])[NPR], double(*restrict d1[NB_LOCAL])[NPR], double(*restrict d2[NB_LOCAL])[NPR], double(*restrict d3[NB_LOCAL])[NPR]){
	int i, j, z, i1, j1, z1, k, i_1, j_1, z_1, i_2, j_2, z_2;
	double U[NPR], U0[NPR], factor[NPR];
	struct of_geom geom;
	struct of_state q;
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, i_1, j_1, z_1, i_2, j_2, z_2,geom,q,U,U0,factor)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, BS_1 - 1, 0, BS_2 - 1, 0, BS_3 - 1) {
			i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * REF_1;
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * REF_2;
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * REF_3;
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d2[nl[n]][index_3D(n, i1, j1 + REF_2, z1)][k] - 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d2[nl[n]][index_3D(n, i1, j1 + REF_2, z1)][k] + 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d1[nl[n]][index_3D(n, i1 + REF_1, j1, z1)][k] - 0.25 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d1[nl[n]][index_3D(n, i1 + REF_1, j1, z1)][k] - 0.25 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d1[nl[n]][index_3D(n, i1 + REF_1, j1, z1)][k] + 0.25 * d2[nl[n]][index_3D(n, i1, j1 + REF_2, z1)][k] - 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * d1[nl[n]][index_3D(n, i1 + REF_1, j1, z1)][k] + 0.25 * d2[nl[n]][index_3D(n, i1, j1 + REF_2, z1)][k] + 0.25 * d3[nl[n]][index_3D(n, i1, j1, z1 + REF_3)][k];
				}
			}
			//Enforce strict conservation of conservative quantitites during refinement
			/*if (i % (1 + REF_1) == REF_1 && j % (1 + REF_2) == REF_2 && z % (1 + REF_3) == REF_3){
				i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * REF_1;
				j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * REF_2;
				z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * REF_3;

				i_1 = N1_GPU_offset[n_child] + i - REF_1;
				i_2 = N1_GPU_offset[n_child] + i;
				j_1 = N2_GPU_offset[n_child] + j - REF_2;
				j_2 = N2_GPU_offset[n_child] + j;
				z_1 = N3_GPU_offset[n_child] + z - REF_3;
				z_2 = N3_GPU_offset[n_child] + z;

				//Calculate total conserved quantities in new cells
				get_geometry(n_child, i_1, j_1, z_1, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)]);
				PLOOP U[k] = 0.125*(dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)][k]);

				get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)][k]);

				get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)][k]);

				get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)][k]);

				get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)][k]);

				get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)][k]);

				get_geometry(n_child, i_2, j_2, z_1, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)][k]);

				get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
				get_state(prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)], &geom, &q);
				primtoU(prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)], &q, &geom, dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)]);
				PLOOP U[k] += 0.125*(dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)][k]);

				//Calculate conserved quantities in old cell
				get_geometry(n, i1, j1, z1, CENT, &geom);
				get_state(prim[nl[n]][index_3D(n, i1, j1, z1)], &geom, &q);
				primtoU(prim[nl[n]][index_3D(n, i1, j1, z1)], &q, &geom, dq[nl[n]][index_3D(n, i1, j1, z1)]);
				PLOOP U0[k] = (dq[nl[n]][index_3D(n, i1, j1, z1)][k]);

				//Normalize new conserved quantities in children to parent cell and convert back to primitive variables
				PLOOP factor[k] = U0[k] / U[k];
				
				PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)][k] *= factor[k];
				get_geometry(n_child, i_1, j_1, z_1, CENT, &geom);
				pflag[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_1)]);
				if (REF_3 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]=Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]);
				}
				if (REF_2 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)]);
				}
				if (REF_2==1 && REF_3 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)]);
				}
				if (REF_1 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)]);
				}
				if (REF_1 == 1 && REF_3==1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)]);
				}
				if (REF_1 == 1 && REF_2 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_2, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)]);
				}
				if (REF_1 == 1 && REF_2 == 1 && REF_3 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)][k] *= factor[k];
					get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_2)]);
				}
			}*/
		}
	}
}


void refine_field(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict pb[NB_LOCAL])[NDIM]){
	int i, j, z, i1, j1, z1, k, ind0, ind2, n_rec1, n_rec2, n_rec3, n_rec4, n_rec5, n_rec6, isize, jsize, zsize;
	double b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8;
	double b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8;
	double b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8;
	int set_1, set_2,set_3,set_4,set_5,set_6;
	double *pointer1, *pointer2, *pointer3, *pointer4, *pointer5, *pointer6;
	//Use divergence free prolongation to handle boundaries
	n_rec1 = -1;
	n_rec2 = -1;
	n_rec3 = -1;
	n_rec4 = -1;
	n_rec5 = -1;
	n_rec6 = -1;

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_5[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_6[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_7[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_8[nl[n]]; n_rec4 = 1; }

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_1[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_2[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_3[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_4[nl[n]]; n_rec2 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_1[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_2[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_5[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_6[nl[n]]; n_rec3 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_3[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_4[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_7[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_8[nl[n]]; n_rec1 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_1[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_3[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_5[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_7[nl[n]]; n_rec5 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_2[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_4[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_6[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_8[nl[n]]; n_rec6 = 1; }
	
	isize = BS_1;
	jsize = BS_2;
	zsize = BS_3;
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, ind0, ind2,b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8,b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8,set_1, set_2,set_3,set_4,set_5,set_6)
	{
	#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, BS_1 - 1 + D1, 0, BS_2 - 1 + D2, 0, BS_3 - 1 + D3) {
			//indices in coarse grid depending on offset (input parameter)
			i1 = (i - i % (1 + REF_1)) / (1 + REF_1);
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2);
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3);

			//index of child
			ind0 = index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);
			ind2 = index_2D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);

			//face centered magnetic field components of neighbouring cells that are refined allready
			b1_1 = b1_2 = b1_3 = b1_4 = b1_5 = b1_6 = b1_7 = b1_8 = 0.;
			b2_1 = b2_2 = b2_3 = b2_4 = b2_5 = b2_6 = b2_7 = b2_8 = 0.;
			b3_1 = b3_2 = b3_3 = b3_4 = b3_5 = b3_6 = b3_7 = b3_8 = 0.;
			set_1 = set_2 = set_3 = set_4 = set_5 = set_6 = -1;
			if ((i == 0 || i == REF_1) && offset_1 == 0 && n_rec2 >= 0){
				b1_1 = pointer2[j1*(1 + REF_2)*zsize + z1*(1 + REF_3)];
				b1_2 = pointer2[j1*(1 + REF_2)*zsize + (z1*(1 + REF_3) + REF_3)];
				b1_3 = pointer2[(j1*(1 + REF_2) + REF_2)*zsize + z1*(1 + REF_3)];
				b1_4 = pointer2[(j1*(1 + REF_2) + REF_2)*zsize + (z1*(1 + REF_3) + REF_3)];
				set_2 = 1;
			}
			else set_2 = -1;
			if ((i == BS_1 || i == BS_1 - REF_1 || i == BS_1 - (1 + REF_1)) && (offset_1 == 1 || REF_1 == 0) && n_rec4 >= 0){
				b1_5 = pointer4[j1*(1 + REF_2)*zsize + z1*(1 + REF_3)];
				b1_6 = pointer4[j1*(1 + REF_2)*zsize + (z1*(1 + REF_3) + REF_3)];
				b1_7 = pointer4[(j1*(1 + REF_2) + REF_2)*zsize + z1*(1 + REF_3)];
				b1_8 = pointer4[(j1*(1 + REF_2) + REF_2)*zsize + (z1*(1 + REF_3) + REF_3)];
				set_4 = 1;
			}
			else set_4 = -1;

			if ((j == 0 || j == REF_2) && offset_2 == 0 && n_rec3 >= 0){
				b2_1 = pointer3[i1*(1 + REF_1)*zsize + z1*(1 + REF_3)];
				b2_2 = pointer3[i1*(1 + REF_1)*zsize + (z1*(1 + REF_3) + REF_3)];
				b2_5 = pointer3[(i1*(1 + REF_1) + REF_1)*zsize + z1*(1 + REF_3)];
				b2_6 = pointer3[(i1*(1 + REF_1) + REF_1)*zsize + (z1*(1 + REF_3) + REF_3)];
				set_3 = 1;
			}
			else set_3 = -1;
			if ((j == BS_2 || j == BS_2 - REF_2 || j == BS_2 - (1 + REF_2)) && (offset_2 == 1 || REF_2 == 0) && n_rec1 >= 0){
				b2_3 = pointer1[i1*(1 + REF_1)*zsize + z1*(1 + REF_3)];
				b2_4 = pointer1[i1*(1 + REF_1)*zsize + (z1*(1 + REF_3) + REF_3)];
				b2_7 = pointer1[(i1*(1 + REF_1) + REF_1)*zsize + z1*(1 + REF_3)];
				b2_8 = pointer1[(i1*(1 + REF_1) + REF_1)*zsize + (z1*(1 + REF_3) + REF_3)];
				set_1 = 1;
			}
			else set_1 = -1;

			if ((z == 0 || z == REF_3) && offset_3 == 0 && n_rec5 >= 0){
				b3_1 = pointer5[i1*(1 + REF_1)*jsize + j1*(1 + REF_2)];
				b3_3 = pointer5[i1*(1 + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				b3_5 = pointer5[(i1*(1 + REF_1) + REF_1)*jsize + j1*(1 + REF_2)];
				b3_7 = pointer5[(i1*(1 + REF_1) + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				set_5 = 1;
			}
			else set_5 = -1;
			if ((z == BS_3 || z == BS_3 - REF_3 || z == BS_3 - (1 + REF_3)) && (offset_3 == 1 || REF_3 == 0) && n_rec6 >= 0){
				b3_2 = pointer6[i1*(1 + REF_1)*jsize + j1*(1 + REF_2)];
				b3_4 = pointer6[i1*(1 + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				b3_6 = pointer6[(i1*(1 + REF_1) + REF_1)*jsize + j1*(1 + REF_2)];
				b3_8 = pointer6[(i1*(1 + REF_1) + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				set_6 = 1;
			}
			else set_6 = -1;

			i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * REF_1 - (i == BS_1 && (offset_1 == 1 || REF_1 == 0));
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * REF_2 - (j == BS_2 && (offset_2 == 1 || REF_2 == 0));
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * REF_3 - (z == BS_3 && (offset_3 == 1 || REF_3 == 0));

			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || REF_1 == 0)), -0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || REF_2 == 0)), -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, -0.25*REF_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}

			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || REF_1 == 0)), -0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || REF_2 == 0)), 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, -0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || REF_1 == 0)), 0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, 0.0, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, 0.25*REF_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || REF_1 == 0)), 0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, 0.0, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, 0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || REF_2 == 0)), -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, -0.25*REF_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || REF_2 == 0)), 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, -0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, 0.0, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, 0.25*REF_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, 0.0, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, 0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
		}
	}
}

void pre_refine(void){
	int n1, i, j, z;

	for (n1 = 0; n1 < n_active; n1++){
		#if(GPU_ENABLED || GPU_DEBUG )
		//GPU_read(n_ord[n1]);
		#endif
		#pragma omp parallel private(i, j, z)
		{
			#pragma omp for collapse(2) schedule(dynamic)
			ZSLOOP3D(N1_GPU_offset[n_ord[n1]] - N1G, N1_GPU_offset[n_ord[n1]] + BS_1 + N1G - 1, -N2G + N2_GPU_offset[n_ord[n1]], N2_GPU_offset[n_ord[n1]] + BS_2 + N2G - 1, N3_GPU_offset[n_ord[n1]] - N3G, N3_GPU_offset[n_ord[n1]] + BS_3 + N3G - 1) {
				psh[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][1] = ps[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][1] * gdet[nl[n_ord[n1]]][index_2D(n_ord[n1], i, j, z)][FACE1];
				psh[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][2] = ps[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][2] * gdet[nl[n_ord[n1]]][index_2D(n_ord[n1], i, j, z)][FACE2];
				psh[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][3] = ps[nl[n_ord[n1]]][index_3D(n_ord[n1], i, j, z)][3] * gdet[nl[n_ord[n1]]][index_2D(n_ord[n1], i, j, z)][FACE3];
			}
		}
	}

	reduce_timestep = 0;
	gpu = 0;
	rc = 0;
	for (n1 = 0; n1 < n_active; n1++)Bp_send1(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec1(n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_send2(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec2(n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_send3(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec3(n_ord[n1]);
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB_AMR \n");
}

void refine(int n){
	int i, j, z, k, n_child, i1, j1, z1, n1;
	//MPI_Barrier(mpi_cartcomm);
	if(rank==0) fprintf(stderr, "Refining block %d %d %d %d \n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
	check_nesting(n); //First make sure nesting criteria are satisfied

	if (rank == 0) if (block[n][AMR_ACTIVE] != 1) fprintf(stderr,"Error trying to refine non-active block %d \n", n);
	
	if (block[n][AMR_NODE] == rank){
		//Calculate gradients, store in flux array F1, F2, F3
		#pragma omp parallel private(i, j, z,k)
		{
			#pragma omp for collapse(2) schedule(dynamic)
			ZSLOOP3D(-D1, BS_1 - 1 + D1, -D2, BS_2 - 1 + D2, -D3, BS_3 - 1 + D3) {
				PLOOP{
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n] - D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n] + D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] - D2, z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + D2, z + N3_GPU_offset[n])][k]);
					#if(N3>1)
					F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] - D3)][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + D3)][k]);
					#endif
				}
			}
		}

		if (block[n][AMR_CHILD1] >= 0){
			set_arrays(block[n][AMR_CHILD1]);
			set_grid(block[n][AMR_CHILD1]);
			n_child = block[n][AMR_CHILD1];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 0, 0, 0, p, F1, F2, F3);
			refine_field(n, n_child, 0, 0, 0, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD1], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD1]);
			#endif
		}

		if (block[n][AMR_CHILD2] >= 0 && REF_3 == 1){
			set_arrays(block[n][AMR_CHILD2]);
			set_grid(block[n][AMR_CHILD2]);
			n_child = block[n][AMR_CHILD2];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 0, 0, 1, p, F1, F2, F3);
			refine_field(n, n_child, 0, 0, 1, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD2], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD2]);
			#endif
		}
		if (block[n][AMR_CHILD3] >= 0 && REF_2 == 1){
			set_arrays(block[n][AMR_CHILD3]);
			set_grid(block[n][AMR_CHILD3]);
			n_child = block[n][AMR_CHILD3];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 0, 1, 0, p, F1, F2, F3);
			refine_field(n, n_child, 0, 1, 0, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD3], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD3]);
			#endif
		}
		if (block[n][AMR_CHILD4] >= 0 && REF_2 == 1 && REF_3 == 1){
			set_arrays(block[n][AMR_CHILD4]);
			set_grid(block[n][AMR_CHILD4]);
			n_child = block[n][AMR_CHILD4];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 0, 1, 1, p, F1, F2, F3);
			refine_field(n, n_child, 0, 1, 1, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD4], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD4]);
			#endif
		}

		if (block[n][AMR_CHILD5] >= 0 && REF_1 == 1){
			set_arrays(block[n][AMR_CHILD5]);
			set_grid(block[n][AMR_CHILD5]);
			n_child = block[n][AMR_CHILD5];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 1, 0, 0, p, F1, F2, F3);
			refine_field(n, n_child, 1, 0, 0, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD5], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD5]);
			#endif
		}
		if (block[n][AMR_CHILD6] >= 0 && REF_1 == 1 && REF_3 == 1){
			set_arrays(block[n][AMR_CHILD6]);
			set_grid(block[n][AMR_CHILD6]);
			n_child = block[n][AMR_CHILD6];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 1, 0, 1, p, F1, F2, F3);
			refine_field(n, n_child, 1, 0, 1, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD6], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD6]);
			#endif
		}
		if (block[n][AMR_CHILD7] >= 0 && REF_1 == 1 && REF_2 == 1){
			set_arrays(block[n][AMR_CHILD7]);
			set_grid(block[n][AMR_CHILD7]);
			n_child = block[n][AMR_CHILD7];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 1, 1, 0, p, F1, F2, F3);
			refine_field(n, n_child, 1, 1, 0, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD7], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD7]);
			#endif
		}
		if (block[n][AMR_CHILD8] >= 0 && REF_1 == 1 && REF_2 == 1 && REF_3 == 1){
			set_arrays(block[n][AMR_CHILD8]);
			set_grid(block[n][AMR_CHILD8]);
			n_child = block[n][AMR_CHILD8];
			block[n_child][AMR_ACTIVE] = 1;
			block[n_child][AMR_NODE] = rank;
			refine_cell(n, n_child, 1, 1, 1, p, F1, F2, F3);
			refine_field(n, n_child, 1, 1, 1, ps);
			#if(GPU_ENABLED || GPU_DEBUG )
			set_arrays_GPU(block[n][AMR_CHILD8], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD8]);
			#endif
		}

	//Clean up memory of parent block
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_finish(n);
	#endif
	free_arrays(n);
}
	//MPI_Barrier(mpi_cartcomm);
	//Take note that block becomes refined
	block[n][AMR_ACTIVE] = 0;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
		block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL];
		if (block[n][AMR_TIMELEVEL] >= 2)block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL] / 2;
		else reduce_timestep = 1;
		block[block[n][i]][AMR_NODE] = block[n][AMR_NODE];
		block[block[n][i]][AMR_ACTIVE] = 1;
	}
	block[n][AMR_TIMELEVEL] = 1;
}

void post_refine(void){
	int n;
	//Allocate memory for all active blocks
	activate_blocks();

	set_corners();
	if (reduce_timestep == 1) dt /= 2.;
	reduce_timestep = 0;
	//Set boundary conditions
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	#endif
}

//Checks if neighbouring blocks are sufficiently refined so that no double jumps in refinement level are created
void check_nesting(int n){
	int i,z;
	for (i = AMR_NBR1; i <= AMR_CORN12; i++){
		if (block[n][i] >= 0 && block[block[n][i]][AMR_PARENT] >= 0 && block[block[block[n][i]][AMR_PARENT]][AMR_ACTIVE] == 1){
			refine(block[block[n][i]][AMR_PARENT]);
		}
	}
}

#if WHICHPROBLEM==DISRUPTION_PROBLEM
#define DENSITY_CUTOFF 0.0000001
#else
#define DENSITY_CUTOFF 16.0
#endif

//Refine on basis of some criteria rhomax (not necessary to use rho though, can also be something different)
void check_refcrit(void){
	int n, task, i,j,z,k, l, level, number;
	int node, n_send, n_blocks;
	double rhomax[NB], rho_rec;
	int NODE[NB];
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	if (max_levels == 0) max_levels = N_LEVELS;
	int tag;
	int count;
	int begin1, end1;
	int one_block_refined = 0;
	if(rank==0) fprintf(stderr,"Starting refinement! \n");

	//First close dump files in progress
	int u;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	if (first_dump == 1){
		for (n = 0; n < n_active; n++){
			MPI_Wait(&req_block[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
			if (dump_cnt % 1 == 0){
				MPI_Wait(&req_blockdiag[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][1]);
			}
		}

		for (u = 0; u < u_max; u++){
			MPI_File_close(&fdump[u]);
			if (dump_cnt % 1 == 0){
				MPI_File_close(&fdumpdiag[u]);
			}
		}
	}
	first_dump = 0;

	//First close rdump files in progress
	if (first_rdump == 1){
		for (n = 0; n < n_active; n++){
			MPI_Wait(&req_block_rdump[nl[n_ord[n]]][0], &Statbound[nl[n_ord[n]]][0]);
			MPI_File_close(&rdump[nl[n_ord[n]]]);
		}
	}
	first_rdump = 0;
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][GDUMP_WRITTEN] == 2){
			if (block[n_ord_total[n]][AMR_NODE] == rank){
				MPI_Wait(&req_gdump1[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_Wait(&req_gdump2[nl[n_ord_total[n]]][0], &Statbound[nl[n_ord_total[n]]][1]);
				MPI_File_close(&gdump[nl[n_ord_total[n]]]);
			}
			block[n_ord_total[n]][GDUMP_WRITTEN] = 1;
		}
	}

	MPI_Barrier(mpi_cartcomm);
	if (rank == 0) fprintf(stderr, "Starting refinement! \n");

	begin1 = time(NULL);

	do{
		tag = 0;
		count = 0;
		n_blocks = n_active_total;

		/*Only allow refinement for one block per node per step*/
		for (i = 0; i < numtasks; i++){
			NODE[i] = 0;
		}

		//Count the number of blocks per node and reset tag
		for (n = 0; n < n_active_total; n++){
			NODE[block[n_ord_total[n]][AMR_NODE]] += 1;
			block[n_ord_total[n]][AMR_TAG] = 0;
		}

		/*First make sure all nodes have the same rhomax*/
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
				rhomax[n_ord_total[n]] = calc_rhomax(n_ord_total[n]);
				for (task = 0; task < numtasks; task++){
					if (rank != task){
						rc = MPI_Isend(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, task,n_active_total+ n % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
						MPI_Request_free(&req[0]);
					}
				}
			}
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				rc = MPI_Irecv(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], n_active_total + n % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
			}
		}
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
			}
		}

		//Tag for refinement
		for (n = 0; n < n_active_total; n++){
			if (rhomax[n_ord_total[n]] > DENSITY_CUTOFF && block[n_ord_total[n]][AMR_LEVEL] < max_levels - 1 && block[n_ord_total[n]][AMR_ACTIVE] == 1){ //If satisfy refinement criterion and smaller than maximum levels
				block[n_ord_total[n]][AMR_TAG] = 1;

				//Refine one level less near black hole
				level = block[n_ord_total[n]][AMR_LEVEL];
				if (block[n_ord_total[n]][AMR_LEVEL] == max_levels - 2 && block[n_ord_total[n]][AMR_COORD1] == 0){
					block[n_ord_total[n]][AMR_TAG] = 0;
				}

				//Do not refine around both poles
				number = 0;
				if (level == 0) number = 0;
				if (level == 1) number = 2;
				if (level == 2) number = 6;
				if (level == 3) number = 14;
				if (level == 4) number = 30;
				if (REF_2 == 0) number = level;
				if ((block[n_ord_total[n]][AMR_COORD2] <= number || block[n_ord_total[n]][AMR_COORD2] >= NB_2*pow(1 + REF_2, level) - number - 1)){
					block[n_ord_total[n]][AMR_TAG] = 0;
				}

				//Do not refine more than one block per NODE
				if (NODE[block[n_ord_total[n]][AMR_NODE]] >= MAX_BLOCKS){
					block[n_ord_total[n]][AMR_TAG] = 0;
					tag = 1;
				}

				if (block[n_ord_total[n]][AMR_TAG] == 1){
					if (one_block_refined == 0) pre_refine();
					one_block_refined = 1;
					//First satisfy nesting criteria
					for (i = AMR_NBR1; i <= AMR_CORN12; i++){
						if (block[n_ord_total[n]][i] >= 0 && block[block[n_ord_total[n]][i]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][i]][AMR_PARENT]][AMR_ACTIVE] == 1){
							if (NODE[block[block[block[n_ord_total[n]][i]][AMR_PARENT]][AMR_NODE]] >= MAX_BLOCKS){
								tag = 1;
								break;
							}
							refine(block[block[n_ord_total[n]][i]][AMR_PARENT]);
							count++;
							NODE[block[block[block[n_ord_total[n]][i]][AMR_PARENT]][AMR_NODE]] += 7;
						}
					}
					//Only refine if nesting criteria satisfied
					if (i == AMR_CORN12 + 1 && NODE[block[n_ord_total[n]][AMR_NODE]] < MAX_BLOCKS){
						one_block_refined = 1;
						refine(n_ord_total[n]);
						count++;
						NODE[block[n_ord_total[n]][AMR_NODE]] += 7;
					}
				}
			}
		}
		//MPI_Barrier(mpi_cartcomm);
		if(one_block_refined==1) post_refine();
		if (tag != 0 && n_active_total<numtasks*MAX_BLOCKS && count>0){
			one_block_refined = 1;
			balance_load();
			#if(GPU_ENABLED)
			balance_load_gpu();
			#endif
			pre_refine();
		}
	} while (tag != 0 && n_active_total<numtasks*MAX_BLOCKS && count>0);

	if (tag==1 && count==0 && rank==0) fprintf(stderr, "Maximum number of blocks exceeded, refinement capped so refinement criterion can not anymore be honoured by H-AMR. Please select more nodes or adjust refinement criterion! \n");

	count = 0;
	
	//First make sure all nodes have the same rhomax
	if (one_block_refined == 1){
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
				rhomax[n_ord_total[n]] = calc_rhomax(n_ord_total[n]);
				for (task = 0; task < numtasks; task++){
					if (rank != task){
						rc = MPI_Isend(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, task, n_active_total + n % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
						MPI_Request_free(&req[0]);
					}
				}
			}
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				rc = MPI_Irecv(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], n_active_total + n % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
			}
		}
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
			}
		}
	}

	int one_block_derefined = 0;
	for (n = 0; n < n_active_total; n++){
		//derefine
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[n_ord_total[n]][AMR_LEVEL]>0){
			block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = -1; //Tag for derefinement

			for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
				if (block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_REFINED] == 1)block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1;
				if (rhomax[block[block[n_ord_total[n]][AMR_PARENT]][i]] > 0.5*DENSITY_CUTOFF) block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1; //Except if one of the children does satisfy the refinement criterion
				if (block[n_ord_total[n]][AMR_LEVEL] > max_levels - 1)block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = -1; //Force derefinement if number of levels reduced due to memory limit problem
			}
		}
	}
	
	for(l=0;l<max_levels;l++){
		for (n = 0; n < n_active_total; n++){
			//do not derefine if required for proper nesting
			for (i = AMR_NBR1; i <= AMR_CORN12; i++){
				if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[n_ord_total[n]][i] >= 0 && block[block[n_ord_total[n]][i]][AMR_TAG] >= 1){
					block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 2;
				}
			}
		}
	}

	//Derefine if tagged for derefinement and not part of nesting
	for (n = 0; n < n_active_total; n++){
		node = block[n_ord_total[n]][AMR_NODE];
		//First make sure all blocks needed for derefinement are on the same node are on the same node
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
			//Send block using non-blocking send
			for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - REF_3)){
				n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
				if (block[n_send][AMR_NODE] != node){
					//if (rank==0)fprintf(stderr,"check_refcrit %d %d %d \n ", block[n_send][AMR_NODE],node, rank);
					rc = 0;
					if (block[n_send][AMR_NODE] == rank){
						rc += MPI_Isend(&p[nl[n_send]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (50 * n_active_total + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][598]);
						#if STAGGERED
						rc += MPI_Isend(&ps[nl[n_send]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (51 * n_active_total + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][597]);
						#endif
					}
					if (rc != 0)fprintf(stderr, "Error in MPI in derefine \n");
				}
			}
		}
	}
	for (n = 0; n < n_active_total; n++){
		node = block[n_ord_total[n]][AMR_NODE];
		//First make sure all blocks needed for derefinement are on the same node are on the same node
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
			//Send block using non-blocking send
			for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - REF_3)){
				n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
				if (block[n_send][AMR_NODE] != node){
					//if (rank==0)fprintf(stderr,"check_refcrit %d %d %d \n ", block[n_send][AMR_NODE],node, rank);
					rc = 0;
					if (node == rank){
						//Allocate memory for active blocks on node
						//set_arrays(n_send);
						//set_grid(n_send);
						temp_p[n_send] = (double(*)[NPR])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NPR]));
						temp_ps[n_send] = (double(*)[NDIM])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NDIM]));
						if (block[n_send][AMR_NODE] >= 0){
							rc += MPI_Irecv(&temp_p[n_send][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (50 * n_active_total + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp1[n_send]);
							#if STAGGERED
							rc += MPI_Irecv(&temp_ps[n_send][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (51 * n_active_total + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp2[n_send]);
							#endif
						}
					}
					if (rc != 0)fprintf(stderr, "Error in MPI in derefine \n");
				}
			}
		}
	}
	for (n = 0; n < n_active_total; n++){
		node = block[n_ord_total[n]][AMR_NODE];
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
			for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - REF_3)){
				//Then use MPI_wait to clean up data that has been sent
				n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
				if (block[n_send][AMR_NODE] != node){
					if (block[n_send][AMR_NODE] == rank){
						MPI_Wait(&boundreqs[nl[n_send]][598], &Statbound[nl[n_send]][0]);
						#if STAGGERED
						MPI_Wait(&boundreqs[nl[n_send]][597], &Statbound[nl[n_send]][1]);
						#endif
						#if(GPU_ENABLED || GPU_DEBUG )
						GPU_finish(n_send);
						#endif
						free_arrays(n_send);
					}
				}
			}
		}
	}
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
			node = block[n_ord_total[n]][AMR_NODE];
			for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - REF_3)){
				//Then initialize sent data on receiving node
				n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
				if (block[n_send][AMR_NODE] != node){
					if (node == rank){
						if (block[n_send][AMR_NODE] >= 0){
							MPI_Wait(&boundreqstemp1[n_send], &Statbound[0][10]);
							#if STAGGERED
							MPI_Wait(&boundreqstemp2[n_send], &Statbound[0][11]);
							#endif
						}
						set_arrays(n_send);
						set_grid(n_send);
						#pragma omp parallel for schedule(dynamic,1)  private(i, j, z, k)
						ZSLOOP3D(N1_GPU_offset[n_send] - N1G, N1_GPU_offset[n_send] + BS_1 - 1 + N1G, N2_GPU_offset[n_send] - N2G, N2_GPU_offset[n_send] + BS_2 - 1 + N2G, N3_GPU_offset[n_send] - N3G, N3_GPU_offset[n_send] + BS_3 - 1 + N3G){
							PLOOP p[nl[n_send]][index_3D(n_send, i, j, z)][k] = temp_p[n_send][index_3D(n_send, i, j, z)][k];
							for (k = 0; k<NDIM; k++) ps[nl[n_send]][index_3D(n_send, i, j, z)][k] = temp_ps[n_send][index_3D(n_send, i, j, z)][k];
						}
						free(temp_p[n_send]);
						free(temp_ps[n_send]);
						#if(GPU_ENABLED || GPU_DEBUG )
						set_arrays_GPU(n_send, block[n_send][AMR_GPU]);
						GPU_write(n_send);
						#endif
					}
				}
			}
		}
	}
	for (n = 0; n < n_active_total; n ++){
		if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
			node = block[n_ord_total[n]][AMR_NODE];
			for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - REF_3)){
				n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
				block[n_send][AMR_NODE] = node;
			}

			block[block[n_ord_total[n]][AMR_PARENT]][AMR_NODE] = node;

			//Then derefine and set corresponding tag and timelevel
			one_block_derefined = 1;
			derefine(block[n_ord_total[n]][AMR_PARENT]);
			count++;
			block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 0;
			//block[block[n_ord_total[n]][AMR_PARENT]][AMR_TIMELEVEL] = block[block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1]][AMR_TIMELEVEL];
		}
	}

	if (one_block_derefined == 1)post_refine();

	balance_load();
	#if(GPU_ENABLED)
	balance_load_gpu();
	#endif
	
	//Set timelevel communicator
	int min_timelevel[8];
	for (i = 0; i <= log(AMR_MAXTIMELEVEL) / log(2); i++){
		if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Comm_free(&row_comm[i]);

		min_timelevel[i] = rank + 1000;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_TIMELEVEL] <= pow(2, i)) min_timelevel[i] = 1;
		}
		MPI_Comm_split(mpi_cartcomm, min_timelevel[i], rank, &row_comm[i]);
	}

	MPI_Barrier(mpi_cartcomm);
	end1 = time(NULL);
	if (rank == 0) fprintf(stderr, "Runtime load balance: %f \n", (double)(end1 - begin1));

	//Start very conservatively
	dt /= 2.;
	for (n = 0; n < n_active_total; n ++){
		block[n_ord_total[n]][AMR_TIMELEVEL] = 1;
	}
}

//This function derefines in z near the pole
int derefine_pole(void){
	int i, j, z, l, ni, nj, nz;
	if (REF_3 != 1 || REF_1 == 1 || REF_2 == 1){
		if(rank==0)fprintf(stderr, "Error! Derefinement near the pole works only for REF_1=0, REF_2=0, REF_3=1 \n");
		exit(20);
		return -1;
	}
	if (NB_2 % 6 != 0){
		if (rank == 0)fprintf(stderr, "For derefinement near the pole chose NB_2 6, 12, 24,48 for 1, 2, 3, 4 levels of derefinement near the pole! \n");
		exit(20);
		return -1;
	}
	if (calc_mem(NB_1*NB_2*NB_3*pow(2., N_LEVELS - 1)) > ((double)numtasks*(double)(numdevices)* 4. * (pow(10., 9.))) && rank == 1) fprintf(stderr, "You are exceeding the maximum memory size of 4 GB per GPU by refining too many blocks! Code will probably segfault, choose a bigger cluster \n");

	for (l = 0; l < N_LEVELS - 1; l++){
		pre_refine();
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3, l);
		for (i = 0; i < ni; i++)for (j = pow(2, l); j < nj - (pow(2, l)); j++)for (z = 0; z < nz; z++){
			if ((double)pow(2, l) < 0.25*NB_2){
				refine(AMR_coord_linear(l, i, j, z));
			}
		}
		MPI_Barrier(mpi_cartcomm);
		post_refine();
		if (rank == 0)fprintf(stderr, "Derefinement at level %d complete! \n", l);
	}
	balance_load();
	#if(GPU_ENABLED)
	balance_load_gpu();
	#endif
	
	return 1;
}

//Set row major order in case of derfinement near pole
void rm_order(void){
	int l, i, j, z, ni, nj, nz;
	int number = 0;
	int number_node = 0;
	ni = NB_1*pow(1 + REF_1, N_LEVELS - 1);
	nj = NB_2*pow(1 + REF_2, N_LEVELS - 1);
	nz = NB_3*pow(1 + REF_3, N_LEVELS - 1);
	for (z = 0; z < nz; z++)for (j = 0; j < nj; j++)for (i = 0; i < ni; i++){
		for (l = 0; l<N_LEVELS; l++){
			if (i<NB_1*pow(1 + REF_1, l) && j<NB_2*pow(1 + REF_2, l) && z<NB_3*pow(1 + REF_3, l) && block[AMR_coord_linear(l, i, j, z)][AMR_ACTIVE] == 1){
				n_ord_total_RM[number] = AMR_coord_linear(l, i, j, z);
				number++;
				if (block[AMR_coord_linear(l, i, j, z)][AMR_NODE] == rank){
					n_ord_RM[number_node] = AMR_coord_linear(l, i, j, z);
					number_node++;
				}
				block[AMR_coord_linear(l, i, j, z)][RM_ORDER] = number;
			}
		}
	}
}

void rm_order2(void){
	int n, l, i, j, z, number;
	int counter = 0;
	int counter2 = 0;
	for (n = 0; n <= n_max; n++){
		AMR_coord_cart_RM(n, &l, &i, &j, &z); //transform to cartesian grid coordinates
		number = AMR_coord_linear(l, i, j, z); //transform to normal lineair ordering
		if (block[number][AMR_ACTIVE] == 1){
			n_ord_total_RM[counter] = number;
			counter++;
		}
		if (block[number][AMR_ACTIVE] == 1 && block[number][AMR_NODE] == rank){
			n_ord_RM[counter2] = number;
			counter2++;
		}
	}
}

double calc_rhomax(int n){
	int i, j, z;
	double rhomax = 0.0;
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			if (p[nl[n]][index_3D(n, i, j, z)][RHO] > rhomax) rhomax = p[nl[n]][index_3D(n, i, j, z)][RHO];
		}
	}
	return rhomax;
}

//Calculates RAM requirements in bytes (conservatively)
double calc_mem(int n_blocks){
	return n_blocks*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * 1000;
}
