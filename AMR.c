#include "decs_MPI.h"

int AMR_coord_linear_RM(int level, int i, int j, int z);
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z);
void rm_order2(void);
void AMR_set_coord(void);

void test_AMR(void){
}

int AMR_coord_linear(int l, int i, int j, int z){
	int index, n, offset, L_1DMAX;
	int b2 = j;
	if (b2 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)b2 + 1.0) / log(2.)), N_LEVELS_1D);
	else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - b2) + 1.0) / log(2.)), N_LEVELS_1D);

	if (l != 0){
		fprintf(stderr, "This function only works at the 0-th level for 3D AMR. Please check and disable this comment if not applicable! \n");
		exit(0);
	}

	if (l<0 || b2<0 || i<0 || j<0 || z<0 || i >= NB_1*pow(1 + (l>L_1DMAX)*REF_1, l - L_1DMAX) || j >= NB_2*pow(1 + (l>L_1DMAX)*REF_2, l - L_1DMAX) || z >= NB_3*pow(1 + REF_1, l)){
		n = -1;
	}
	else{
		offset = N_LEVELS_1D - L_1DMAX;
		index = (int)(i * NB_3*(int)pow(1 + REF_3, l + offset) * NB_2*pow(1 + REF_2*((l + offset) > N_LEVELS_1D), l - N_LEVELS_1D + offset) + j * NB_3*pow(1 + REF_3, l + offset) + z);
		n = lin_coord[l + offset][index];
		if (block[n][AMR_LEVEL] != l || block[n][AMR_COORD1] != i || block[n][AMR_COORD2] != j || block[n][AMR_COORD3] != z){
			fprintf(stderr, "Could not find the right linear coordinate, input incorrect! \n");
			fprintf(stderr, "Incorrect values are l: %d j0: %d i: %d j: %d z: %d \n", l, b2, i, j, z);
			exit(0);
		}
	}

	return n;
}

int AMR_coord_linear2(int l, int b2, int i, int j, int z){
	int index, n, offset, L_1DMAX;

	if (b2 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)b2 + 1.0) / log(2.)), N_LEVELS_1D);
	else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - b2) + 1.0) / log(2.)), N_LEVELS_1D);

	if (l<0 || b2<0 || i<0 || j<0 || z<0 || i >= NB_1*pow(1 + (l>L_1DMAX)*REF_1, l - L_1DMAX) || j >= NB_2*pow(1 + (l>L_1DMAX)*REF_2, l - L_1DMAX) || z >= NB_3*pow(1 + REF_1, l)){
		n = -1;
	}
	else{
		offset = N_LEVELS_1D - L_1DMAX;
		index = (int)(i * NB_3*(int)pow(1 + REF_3, l + offset) * NB_2*pow(1 + REF_2*((l + offset) > N_LEVELS_1D), l - N_LEVELS_1D + offset) + j * NB_3*pow(1 + REF_3, l + offset) + z);
		n = lin_coord[l + offset][index];
		if (block[n][AMR_LEVEL] != l || block[n][AMR_COORD1] != i || block[n][AMR_COORD2] != j || block[n][AMR_COORD3] != z){
			fprintf(stderr, "Could not find the right linear coordinate, input incorrect! \n");
			fprintf(stderr, "Incorrect values are l: %d j0: %d i: %d j: %d z: %d \n", l, b2, i, j, z);
			exit(0);
		}
	}

	return n;
}

int AMR_coord_linear_RM(int level, int i, int j, int z){
	int index, n;
	index = (int)(i * NB_3*(int)pow(1 + REF_3, level) * NB_2*pow(1 + REF_2, level) + j * NB_3*pow(1 + REF_3, level) + z);
	n = lin_coord_RM[level][index];

	return n;
}

//Given a certain linear coordinate n this function determines the cartesian coordinates of a block and it's corresponding AMR-level
void AMR_coord_cart(int n, int *level, int *i, int *j, int *z){
	*level = block[n][AMR_LEVEL];
	*i = block[n][AMR_COORD1];
	*j = block[n][AMR_COORD2];
	*z = block[n][AMR_COORD3];
}

//Given a certain linear coordinate n this function determines the cartesian coordinates of a block and it's corresponding AMR-level
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z){
	*level = block[n][AMR_LEVEL];
	*i = block[n][AMR_COORD1];
	*j = block[n][AMR_COORD2];
	*z = block[n][AMR_COORD3];
}

//Given a certain linear coordinate n this function determines the cartesian coordinates of a block and it's corresponding AMR-level
void AMR_set_coord(void){
	int n, l, l_1D, l_3D, L_1DMAX, lc, i[N_LEVELS], j[N_LEVELS], z[N_LEVELS], keep_while, keep_looping, index, offset;

	//Initialize counters
	for (l = 0; l < N_LEVELS; l++) i[l] = j[l] = z[l] = 0;
	l_1D = l_3D = l = 0;
	n = 0;
	keep_while = 1;
	while (keep_while){
		//Set level based on values from last iteration
		block[n][AMR_LEVEL1] = l_3D;
		block[n][AMR_LEVEL2] = l_3D;
		block[n][AMR_LEVEL3] = l_1D + l_3D;
		l = l_1D + l_3D;
		block[n][AMR_LEVEL] = l;
	
		if (!(N_LEVELS_1D == 0 || (NB_2 == 6 != 0 && N_LEVELS_1D == 1) || (NB_2 == 12 != 0 && N_LEVELS_1D == 2) || (NB_2 == 24 != 0 && N_LEVELS_1D == 3) || (NB_2 == 48 != 0 && N_LEVELS_1D == 4) || (NB_2 == 96 != 0 && N_LEVELS_1D == 5))){
			if (rank == 0)fprintf(stderr, "For derefinement near the pole chose NB_2 6, 12, 24, 48, 96 for 1, 2, 3, 4, 5 levels of derefinement near the pole! \n");
			exit(0);
		}

		//Based on value of 0-th level block determine the number of 1D refinement levels
		if (j[0] < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)j[0] + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - j[0]) + 1.0) / log(2.)), N_LEVELS_1D);

		//Set coordinates based on values from last iteration
		block[n][AMR_COORD1] = block[n][AMR_COORD2] = block[n][AMR_COORD3] = 0;
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++)	block[n][AMR_COORD1] += i[lc] * pow(1 + (block[n][AMR_LEVEL] >= L_1DMAX)*REF_1, block[n][AMR_LEVEL] - MY_MAX(lc, L_1DMAX));
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++)	block[n][AMR_COORD2] += j[lc] * pow(1 + (block[n][AMR_LEVEL] >= L_1DMAX)*REF_2, block[n][AMR_LEVEL] - MY_MAX(lc, L_1DMAX));
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++)	block[n][AMR_COORD3] += z[lc] * pow(1 + REF_3, block[n][AMR_LEVEL] - lc);
		
		//Store in array such that one can recover linear coordinate based on 4D coordinate
		offset = N_LEVELS_1D - L_1DMAX;
		index = (int)(block[n][AMR_COORD1] * NB_3*(int)pow(1 + REF_3, l + offset) * NB_2*pow(1 + REF_2*((l + offset) > N_LEVELS_1D), l - N_LEVELS_1D + offset) + block[n][AMR_COORD2] * NB_3*pow(1 + REF_3, l + offset) + block[n][AMR_COORD3]);
		lin_coord[l + offset][index] = n;
		lin_coord_RM[l + offset][index] = n;

		//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0) fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d L_1DMAX: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], L_1DMAX, block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);

		//Break out of loop if maximum block number reached and set n_max
		if (l == N_LEVELS_3D - 1 && block[n][AMR_COORD1] == NB_1*pow(1 + REF_1, l) - 1 && block[n][AMR_COORD2] == NB_2*pow(1 + REF_2, l) - 1 && block[n][AMR_COORD3] == NB_3*pow(1 + REF_3, l) - 1){
			n_max = n;
			break;
		}
		
		//Advance linear index by 1
		if (l < N_LEVELS_3D+L_1DMAX-1){	
			l++;
			i[l] = j[l] = z[l] = 0;
			if (l_1D < L_1DMAX) l_1D++;
			else l_3D++;
		}
		else{
			do{
				keep_looping = 0;
				//At 0-th level use row-major ordering with z fastest running index
				if (l == 0){
					z[l]++;
					if (z[l] == NB_3){
						z[l] = 0;
						j[l]++;
					}
					if (j[l] == NB_2){
						j[l] = 0;
						i[l]++;
					}
					if (i[l] == NB_1){
						fprintf(stderr, "Catastrophic error in grid mapping! \n");
						exit(0);
					}
				}
				else{
					z[l]++;
					if (z[l] == 1 + REF_3){
						z[l] = 0;
						j[l]++;
					}
					if (j[l] == 1 + (l > L_1DMAX)*REF_2){
						j[l] = 0;
						i[l]++;
					}
					if (i[l] == 1 + (l > L_1DMAX)*REF_1){
						l--;
						if (l<0){
							fprintf(stderr, "Negative level in grid mapping! \n");
							exit(0);
						}
						if (l_3D > 0) l_3D--;
						else l_1D--;
						keep_looping = 1;
					}
				}
			} while (keep_looping);
		}
	
		n++;
	}
}


void set_ref(int n, int n_rec, int *ref_1, int *ref_2, int * ref_3){
	ref_1[0] = block[n_rec][AMR_LEVEL1] - block[n][AMR_LEVEL1];
	ref_2[0] = block[n_rec][AMR_LEVEL2] - block[n][AMR_LEVEL2];
	ref_3[0] = block[n_rec][AMR_LEVEL3] - block[n][AMR_LEVEL3];
}

//Sets the AMR hierarchy
void set_AMR(void){
	int n, n_parent, n_child[9], n_nbr[21], level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1,
		i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag;
	int y, rem, node = 0;

	//Allocate arrays that are not block-specific and thus only need to be allocated at the start of a run and not between refinement steps
	block = (int(*)[NV])calloc(NB+1, sizeof(int[NV]));
	for (l = 0; l < N_LEVELS_3D; l++){
		E_avg1[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1+REF_1, l)*NB_3*pow(1+REF_3,l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg2[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg1_new[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg2_new[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));

	}
	max_levels = 0;

	//Set memory flag to unallocated
	for (i = 0; i < NB_LOCAL; i++){
		mem_spot[i] = -1;
		mem_spot_gpu[i] = -1;
	}

	//Create array for linear coordinates
	for (l = 0; l < N_LEVELS; l++){
		lin_coord[l] = (int *)calloc(NB_1*pow(1 + (l > N_LEVELS_1D)*REF_1, (l - N_LEVELS_1D))*NB_2*pow(1 + (l > N_LEVELS_1D)*REF_2, (l - N_LEVELS_1D))*NB_3*pow(1 + REF_3, l), sizeof(int));
		lin_coord_RM[l] = (int *)calloc(NB_1*pow(1 + (l > N_LEVELS_1D)*REF_1, (l - N_LEVELS_1D))*NB_2*pow(1 + (l > N_LEVELS_1D)*REF_2, (l - N_LEVELS_1D))*NB_3*pow(1 + REF_3, l), sizeof(int));
	}

	//Set mapping from linear to 3D coordinates and vice-versa
	AMR_set_coord();

	//Indicates something is wrok with the grid mapping
	if (n_max != NB - 1){
		fprintf(stderr, "n_max: %d and NB: %d do not match! \n", n_max, NB);
		exit(0);
	}

 	//Find parent for all blocks(refined and unrefined)
	for (n = 0; n <= n_max; n++){
		//Find parent of block
		if (block[n][AMR_LEVEL] == 0) block[n][AMR_PARENT] = -1; //-1 means no parent
		else{
			if (block[n][AMR_LEVEL1] > 0 || block[n][AMR_LEVEL2] > 0){
				new_level1 = block[n][AMR_LEVEL1] - REF_1;
				new_level2 = block[n][AMR_LEVEL2] - REF_2;
				new_level3 = block[n][AMR_LEVEL3] - REF_3;
			}
			else if (block[n][AMR_LEVEL3] > 0){
				new_level1 = block[n][AMR_LEVEL1];
				new_level2 = block[n][AMR_LEVEL2];
				new_level3 = block[n][AMR_LEVEL3] - REF_3;
			}
			else{
				fprintf(stderr, "Something went wrong when setting parents for each block! \n");
				exit(0);
			}
			ref_1 = block[n][AMR_LEVEL1] - new_level1;
			ref_2 = block[n][AMR_LEVEL2] - new_level2;
			ref_3 = block[n][AMR_LEVEL3] - new_level3;

			i_parent = (block[n][AMR_COORD1] - block[n][AMR_COORD1] % (1 + ref_1)) / (1 + ref_1);
			j_parent = (block[n][AMR_COORD2] - block[n][AMR_COORD2] % (1 + ref_2)) / (1 + ref_2);
			z_parent = (block[n][AMR_COORD3] - block[n][AMR_COORD3] % (1 + ref_3)) / (1 + ref_3);
			j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
			block[n][AMR_PARENT] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, i_parent, j_parent, z_parent);
			//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
			//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[block[n][AMR_PARENT]][AMR_LEVEL], block[block[n][AMR_PARENT]][AMR_LEVEL1], block[block[n][AMR_PARENT]][AMR_LEVEL2], block[block[n][AMR_PARENT]][AMR_LEVEL3], block[block[n][AMR_PARENT]][AMR_COORD1], block[block[n][AMR_PARENT]][AMR_COORD2], block[block[n][AMR_PARENT]][AMR_COORD3]);

		}
	}

	//Find children of block, -1 means no children
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		if (block[n][AMR_LEVEL] == N_LEVELS_3D + L_1DMAX-1){
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
			
			if (block[n][AMR_LEVEL] < L_1DMAX){
				new_level1 = block[n][AMR_LEVEL1];
				new_level2 = block[n][AMR_LEVEL2];
				new_level3 = block[n][AMR_LEVEL3] + REF_3;
			}
			else if (block[n][AMR_LEVEL] >= L_1DMAX){
				new_level1 = block[n][AMR_LEVEL1] + REF_1;
				new_level2 = block[n][AMR_LEVEL2] + REF_2;
				new_level3 = block[n][AMR_LEVEL3] + REF_3;
			}
			else{
				fprintf(stderr, "Something went wrong when setting children for each block! \n");
				exit(0);
			}
			ref_1 = new_level1 - block[n][AMR_LEVEL1];
			ref_2 = new_level2 - block[n][AMR_LEVEL2];
			ref_3 = new_level3 - block[n][AMR_LEVEL3];
			block[n][AMR_CHILD1] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
			block[n][AMR_CHILD2] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			block[n][AMR_CHILD3] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
			block[n][AMR_CHILD4] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			block[n][AMR_CHILD5] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
			block[n][AMR_CHILD6] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			block[n][AMR_CHILD7] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
			block[n][AMR_CHILD8] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
			//int test = AMR_CHILD1;
			//fprintf(stderr, "NB: %d n_child: %d j0: %d ref_1: %d ref_2: %d ref_3: %d L_1DMAX: %d \n", NB, block[n][test], j0, ref_1,ref_2,ref_3, L_1DMAX);
			//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[block[n][test]][AMR_LEVEL], block[block[n][test]][AMR_LEVEL1], block[block[n][test]][AMR_LEVEL2], block[block[n][test]][AMR_LEVEL3], block[block[n][test]][AMR_COORD1], block[block[n][test]][AMR_COORD2], block[block[n][test]][AMR_COORD3]);		
		}
	}

	//Find neighbours of block
	for (n = 0; n <= n_max; n++){
		//Set maximum coordinates
		i_max = NB_1*pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1;
		j_max = NB_2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1;
		z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;

		i = block[n][AMR_COORD1];
		j = block[n][AMR_COORD2];
		z = block[n][AMR_COORD3];
		level = block[n][AMR_LEVEL];

		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		//First dimension is trivial
		if (i + 1 > i_max && PERIODIC1 == 1) i1 = 0;
		else if (i + 1 > i_max && PERIODIC1 == 0) i1 = -1;
		else i1 = i + 1;
		block[n][AMR_NBR2] = AMR_coord_linear2(level, j0, i1, j, z);

		if (i - 1 < 0 && PERIODIC1 == 1) i1 = i_max;
		else i1 = i - 1;
		block[n][AMR_NBR4] = AMR_coord_linear2(level, j0, i1, j, z);

		//Third dimension is trivial
		if (z + 1 > z_max && PERIODIC3 == 1) z1 = 0;
		else if (z + 1 > z_max && PERIODIC3 == 0) z1 = -1;
		else z1 = z + 1;
		block[n][AMR_NBR5] = AMR_coord_linear2(level, j0, i, j, z1);

		if (z - 1 < 0 && PERIODIC3 == 1) z1 = z_max;
		else z1 = z - 1;
		block[n][AMR_NBR6] = AMR_coord_linear2(level, j0, i, j, z1);

		//Second dimension is not trivial
		if (j - 1 < 0 && PERIODIC2 == 1) j1 = j_max;
		else j1 = j - 1;
		flag = 0;
		for (l = 0; l < MY_MIN(level, N_LEVELS_1D); l++){
			jbound=pow(1 + REF_2, l); 
			if ((j == jbound * pow(1 + REF_2, block[n][AMR_LEVEL2])) || (j == (NB_2  - jbound) * pow(1 + REF_2, block[n][AMR_LEVEL2]))){
				block[n][AMR_NBR1] = NB;
				flag = 1;
				break;
			}

		}
		if (flag == 0) block[n][AMR_NBR1] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z);

		if (j + 1 > j_max && PERIODIC2 == 1) j1 = 0;
		else if (j + 1 > j_max && PERIODIC2 == 0) j1 = -1;
		else j1 = j + 1;
		flag = 0;
		for (l = 0; l < MY_MIN(level, N_LEVELS_1D); l++){
			jbound = pow(1 + REF_2, l);
			if (j == ((NB_2 - jbound)* pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) || (j == jbound* pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1)){
				block[n][AMR_NBR3] = NB;
				//if (i == 0 && z == 0)fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
				flag = 1;
				break;
			}
		}
		if (flag == 0) block[n][AMR_NBR3] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z);

		//Set transmissive pole
		block[n][AMR_POLE] = 0; 
		#if (TRANS_BOUND)
		//if (NB_3 % 2 != 0 && rank==0) fprintf(stderr, "Number of blocks in the third dimension is not an even number. This is incompatible with TRANS_BOUND");
		//Find the neighbours in the case we have transmissive boundary conditions at the pole
		if (j == 0){
			block[n][AMR_POLE] += 1;
			block[n][AMR_NBR1] = AMR_coord_linear2(level, j0, i, j, (z + NB_3*(int)pow(1 + REF_3, level) / 2) % (z_max + 1));
		}
		if (j == j_max){
			block[n][AMR_POLE] += 2;
			block[n][AMR_NBR3] = AMR_coord_linear2(level, j0, i, j, (z + NB_3*(int)pow(1 + REF_3, level) / 2) % (z_max + 1));
		}
		#endif
	}
		
	//Find corners of block assuming only third dimension is periodic
	for (n = 0; n <= n_max; n++){
		//Set maximum coordinates
		i_max = NB_1*pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1;
		j_max = NB_2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1;
		z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;

		i = block[n][AMR_COORD1];
		j = block[n][AMR_COORD2];
		z = block[n][AMR_COORD3];
		level = block[n][AMR_LEVEL];

		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		//x-y plane
		if (i + 1 > i_max || j - 1 < 0){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i + 1;
			j1 = j - 1;
		}
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR2] >= 0) block[n][AMR_CORN1] = NB;
		else block[n][AMR_CORN1] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		if (i + 1 > i_max || j + 1 > j_max){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i + 1;
			j1 = j + 1;
		}
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR2] >= 0) block[n][AMR_CORN2] = NB;
		else block[n][AMR_CORN2] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		if (i - 1 < 0 || j + 1 > j_max){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i - 1;
			j1 = j + 1;
		}
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR4] >= 0) block[n][AMR_CORN3] = NB;
		else block[n][AMR_CORN3] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		if (i - 1 < 0 || j - 1 < 0){
			j1 = -1; i1 = -1;
		}
		else{
			i1 = i - 1;
			j1 = j - 1;
		}
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR4] >= 0) block[n][AMR_CORN4] = NB;
		else block[n][AMR_CORN4] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		//x-z plane
		i1 = i + 1;
		z1 = z - 1;
		if (i + 1 > i_max) i1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN5] = AMR_coord_linear2(level, j0, i1, j, z1);

		i1 = i + 1;
		z1 = z + 1;
		if (i + 1 > i_max) i1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN6] = AMR_coord_linear2(level, j0, i1, j, z1);

		i1 = i - 1;
		z1 = z + 1;
		if (i - 1 < 0) i1 = -1;
		if (z + 1 > z_max) z1 = 0;
		block[n][AMR_CORN7] = AMR_coord_linear2(level, j0, i1, j, z1);


		i1 = i - 1;
		z1 = z - 1;
		if (i - 1 < 0) i1 = -1;
		if (z - 1 < 0) z1 = z_max;
		block[n][AMR_CORN8] = AMR_coord_linear2(level, j0, i1, j, z1);

		//y-z plane
		j1 = j - 1;
		z1 = z + 1;
		if (j - 1 < 0) j1 = -1;
		if (z + 1 > z_max) z1 = 0;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR5] >= 0) block[n][AMR_CORN9] = NB;
		else block[n][AMR_CORN9] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		j1 = j + 1;
		z1 = z + 1;
		if (j + 1 > j_max) j1 = -1;
		if (z + 1 > z_max) z1 = 0;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR5] >= 0) block[n][AMR_CORN10] = NB;
		else block[n][AMR_CORN10] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		j1 = j + 1;
		z1 = z - 1;
		if (j + 1 > j_max) j1 = -1;
		if (z - 1 < 0) z1 = z_max;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR6] >= 0) block[n][AMR_CORN11] = NB;
		else block[n][AMR_CORN11] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		j1 = j - 1;
		z1 = z - 1;
		if (j - 1 < 0) j1 = -1;
		if (z - 1 < 0) z1 = z_max;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR6] >= 0) block[n][AMR_CORN12] = NB;
		else block[n][AMR_CORN12] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);
	}

	//Set 2-way grid to negative
	for (n = 0; n <= n_max; n++) for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) block[n][i] = -1;
	
	//Set 2-way grid hierarchy
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		//Set NBR children
		if (block[n][AMR_NBR1] >= 0){
			if (block[n][AMR_NBR1] != NB){
				block[n][AMR_NBR1_3] = block[block[n][AMR_NBR1]][AMR_CHILD3];
				block[n][AMR_NBR1_4] = block[block[n][AMR_NBR1]][AMR_CHILD4];
				block[n][AMR_NBR1_7] = block[block[n][AMR_NBR1]][AMR_CHILD7];
				block[n][AMR_NBR1_8] = block[block[n][AMR_NBR1]][AMR_CHILD8];
			}
			else{
				//Difference in REF_1, REF_2
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					if (block[n][AMR_LEVEL1] < N_LEVELS_3D - 1){
						block[n][AMR_NBR1_3] = AMR_coord_linear2(block[n][AMR_LEVEL], j0 - 1, block[n][AMR_COORD1] * (1 + REF_1), block[n][AMR_COORD2] * (1 + REF_2) - 1, block[n][AMR_COORD3]);
						block[n][AMR_NBR1_4] = block[n][AMR_NBR1_3];
						block[n][AMR_NBR1_7] = AMR_coord_linear2(block[n][AMR_LEVEL], j0 - 1, block[n][AMR_COORD1] * (1 + REF_1) + REF_1, block[n][AMR_COORD2] * (1 + REF_2) - 1, block[n][AMR_COORD3]);
						block[n][AMR_NBR1_8] = block[n][AMR_NBR1_7];
					}
				}
				else{ //Difference in REF_3
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Orig: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
					block[n][AMR_NBR1_3] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0 - 1, block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] * (1 + REF_3));
					block[n][AMR_NBR1_4] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0 - 1, block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] * (1 + REF_3) + 1);
					block[n][AMR_NBR1_7] = block[n][AMR_NBR1_3];
					block[n][AMR_NBR1_8] = block[n][AMR_NBR1_4];
					//int test = block[n][AMR_NBR1_3];
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Child: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
				}
			}
		}
		if (block[n][AMR_NBR2] >= 0){
			block[n][AMR_NBR2_1] = block[block[n][AMR_NBR2]][AMR_CHILD1];
			block[n][AMR_NBR2_2] = block[block[n][AMR_NBR2]][AMR_CHILD2];
			block[n][AMR_NBR2_3] = block[block[n][AMR_NBR2]][AMR_CHILD3];
			block[n][AMR_NBR2_4] = block[block[n][AMR_NBR2]][AMR_CHILD4];
		}
		if (block[n][AMR_NBR3] >= 0){
			if (block[n][AMR_NBR3] != NB){
				block[n][AMR_NBR3_1] = block[block[n][AMR_NBR3]][AMR_CHILD1];
				block[n][AMR_NBR3_2] = block[block[n][AMR_NBR3]][AMR_CHILD2];
				block[n][AMR_NBR3_5] = block[block[n][AMR_NBR3]][AMR_CHILD5];
				block[n][AMR_NBR3_6] = block[block[n][AMR_NBR3]][AMR_CHILD6];
			}
			else{
				//Difference in REF_3
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Orig: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
					block[n][AMR_NBR3_1] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0 + 1, block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] * (1 + REF_3));
					block[n][AMR_NBR3_2] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0 + 1, block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] * (1 + REF_3) + 1);
					block[n][AMR_NBR3_5] = block[n][AMR_NBR3_1];
					block[n][AMR_NBR3_6] = block[n][AMR_NBR3_2];
					//int test = block[n][AMR_NBR3_1];
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Child: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
				}
				else{ //Difference in REF_1, REF_2
					if (block[n][AMR_LEVEL1] < N_LEVELS_3D - 1){
						block[n][AMR_NBR3_1] = AMR_coord_linear2(block[n][AMR_LEVEL], j0 + 1, block[n][AMR_COORD1] * (1 + REF_1), (block[n][AMR_COORD2] + 1) * (1 + REF_2), block[n][AMR_COORD3]);
						block[n][AMR_NBR3_2] = block[n][AMR_NBR3_1];
						block[n][AMR_NBR3_5] = AMR_coord_linear2(block[n][AMR_LEVEL], j0 + 1, block[n][AMR_COORD1] * (1 + REF_1) + REF_1, (block[n][AMR_COORD2] + 1) * (1 + REF_2), block[n][AMR_COORD3]);
						block[n][AMR_NBR3_6] = block[n][AMR_NBR3_5];
					}
				}
			}
		}
		if (block[n][AMR_NBR4] >= 0){
			block[n][AMR_NBR4_5] = block[block[n][AMR_NBR4]][AMR_CHILD5];
			block[n][AMR_NBR4_6] = block[block[n][AMR_NBR4]][AMR_CHILD6];
			block[n][AMR_NBR4_7] = block[block[n][AMR_NBR4]][AMR_CHILD7];
			block[n][AMR_NBR4_8] = block[block[n][AMR_NBR4]][AMR_CHILD8];
		}
		if (block[n][AMR_NBR5] >= 0){
			block[n][AMR_NBR5_1] = block[block[n][AMR_NBR5]][AMR_CHILD1];
			block[n][AMR_NBR5_3] = block[block[n][AMR_NBR5]][AMR_CHILD3];
			block[n][AMR_NBR5_5] = block[block[n][AMR_NBR5]][AMR_CHILD5];
			block[n][AMR_NBR5_7] = block[block[n][AMR_NBR5]][AMR_CHILD7];
		}
		if (block[n][AMR_NBR6] >= 0){
			block[n][AMR_NBR6_2] = block[block[n][AMR_NBR6]][AMR_CHILD2];
			block[n][AMR_NBR6_4] = block[block[n][AMR_NBR6]][AMR_CHILD4];
			block[n][AMR_NBR6_6] = block[block[n][AMR_NBR6]][AMR_CHILD6];
			block[n][AMR_NBR6_8] = block[block[n][AMR_NBR6]][AMR_CHILD8];
		}
	}

	//Set corn children
	for (n = 0; n <= n_max; n++){
		if (block[n][AMR_CORN1] >= 0){
			if (block[n][AMR_CORN1] != NB){
				block[n][AMR_CORN1_1] = block[block[n][AMR_CORN1]][AMR_CHILD3];
				block[n][AMR_CORN1_2] = block[block[n][AMR_CORN1]][AMR_CHILD4];
			}
			else{
				if (block[n][AMR_NBR1_7] >= 0) block[n][AMR_CORN1_1] = block[block[n][AMR_NBR1_7]][AMR_NBR2];
				if (block[n][AMR_NBR1_8] >= 0) block[n][AMR_CORN1_2] = block[block[n][AMR_NBR1_8]][AMR_NBR2];
			}
		}
		if (block[n][AMR_CORN2] >= 0){
			if (block[n][AMR_CORN2] != NB){
				block[n][AMR_CORN2_1] = block[block[n][AMR_CORN2]][AMR_CHILD1];
				block[n][AMR_CORN2_2] = block[block[n][AMR_CORN2]][AMR_CHILD2];
			}
			else{
				if (block[n][AMR_NBR3_5] >= 0) block[n][AMR_CORN2_1] = block[block[n][AMR_NBR3_5]][AMR_NBR2];
				if (block[n][AMR_NBR3_6] >= 0) block[n][AMR_CORN2_2] = block[block[n][AMR_NBR3_6]][AMR_NBR2];
			}
		}
		if (block[n][AMR_CORN3] >= 0){
			if (block[n][AMR_CORN3] != NB){
				block[n][AMR_CORN3_1] = block[block[n][AMR_CORN3]][AMR_CHILD5];
				block[n][AMR_CORN3_2] = block[block[n][AMR_CORN3]][AMR_CHILD6];
			}
			else{
				if (block[n][AMR_NBR3_1] >= 0) block[n][AMR_CORN3_1] = block[block[n][AMR_NBR3_1]][AMR_NBR4];
				if (block[n][AMR_NBR3_2] >= 0) block[n][AMR_CORN3_2] = block[block[n][AMR_NBR3_2]][AMR_NBR4];
			}
		}
		if (block[n][AMR_CORN4] >= 0){
			if (block[n][AMR_CORN4] != NB){
				block[n][AMR_CORN4_1] = block[block[n][AMR_CORN4]][AMR_CHILD7];
				block[n][AMR_CORN4_2] = block[block[n][AMR_CORN4]][AMR_CHILD8];
			}
			else{
				if (block[n][AMR_NBR1_3] >= 0) block[n][AMR_CORN4_1] = block[block[n][AMR_NBR1_3]][AMR_NBR4];
				if (block[n][AMR_NBR1_4] >= 0) block[n][AMR_CORN4_2] = block[block[n][AMR_NBR1_4]][AMR_NBR4];
			}
		}

		if (block[n][AMR_CORN5] >= 0){
			block[n][AMR_CORN5_1] = block[block[n][AMR_CORN5]][AMR_CHILD2];
			block[n][AMR_CORN5_2] = block[block[n][AMR_CORN5]][AMR_CHILD4];
		}
		if (block[n][AMR_CORN6] >= 0){
			block[n][AMR_CORN6_1] = block[block[n][AMR_CORN6]][AMR_CHILD1];
			block[n][AMR_CORN6_2] = block[block[n][AMR_CORN6]][AMR_CHILD3];
		}
		if (block[n][AMR_CORN7] >= 0){
			block[n][AMR_CORN7_1] = block[block[n][AMR_CORN7]][AMR_CHILD5];
			block[n][AMR_CORN7_2] = block[block[n][AMR_CORN7]][AMR_CHILD7];
		}
		if (block[n][AMR_CORN8] >= 0){
			block[n][AMR_CORN8_1] = block[block[n][AMR_CORN8]][AMR_CHILD6];
			block[n][AMR_CORN8_2] = block[block[n][AMR_CORN8]][AMR_CHILD8];
		}

		if (block[n][AMR_CORN9] >= 0){
			if (block[n][AMR_CORN9] != NB){
				block[n][AMR_CORN9_1] = block[block[n][AMR_CORN9]][AMR_CHILD3];
				block[n][AMR_CORN9_2] = block[block[n][AMR_CORN9]][AMR_CHILD7];
			}
			else{
				if (block[n][AMR_NBR1_4] >= 0) block[n][AMR_CORN9_1] = block[block[n][AMR_NBR1_4]][AMR_NBR5];
				if (block[n][AMR_NBR1_8] >= 0) block[n][AMR_CORN9_2] = block[block[n][AMR_NBR1_8]][AMR_NBR5];
			}
		}
		if (block[n][AMR_CORN10] >= 0){
			if (block[n][AMR_CORN10] != NB){
				block[n][AMR_CORN10_1] = block[block[n][AMR_CORN10]][AMR_CHILD1];
				block[n][AMR_CORN10_2] = block[block[n][AMR_CORN10]][AMR_CHILD5];
			}
			else{
				if (block[n][AMR_NBR3_2] >= 0) block[n][AMR_CORN10_1] = block[block[n][AMR_NBR3_2]][AMR_NBR5];
				if (block[n][AMR_NBR3_6] >= 0) block[n][AMR_CORN10_2] = block[block[n][AMR_NBR3_6]][AMR_NBR5];
			}
		}
		if (block[n][AMR_CORN11] >= 0){
			if (block[n][AMR_CORN11] != NB){
				block[n][AMR_CORN11_1] = block[block[n][AMR_CORN11]][AMR_CHILD2];
				block[n][AMR_CORN11_2] = block[block[n][AMR_CORN11]][AMR_CHILD6];
			}
			else{
				if (block[n][AMR_NBR3_1] >= 0) block[n][AMR_CORN11_1] = block[block[n][AMR_NBR3_1]][AMR_NBR6];
				if (block[n][AMR_NBR3_5] >= 0) block[n][AMR_CORN11_2] = block[block[n][AMR_NBR3_5]][AMR_NBR6];
			}
		}
		if (block[n][AMR_CORN12] >= 0){
			if (block[n][AMR_CORN12] != NB){
				block[n][AMR_CORN12_1] = block[block[n][AMR_CORN12]][AMR_CHILD4];
				block[n][AMR_CORN12_2] = block[block[n][AMR_CORN12]][AMR_CHILD8];
			}
			else{
				if (block[n][AMR_NBR1_3] >= 0) block[n][AMR_CORN12_1] = block[block[n][AMR_NBR1_3]][AMR_NBR6];
				if (block[n][AMR_NBR1_7] >= 0) block[n][AMR_CORN12_2] = block[block[n][AMR_NBR1_7]][AMR_NBR6];
			}
		}
	}

	//Set NBR parent
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));

		if (block[n][AMR_NBR1] >= 0){
			if (block[n][AMR_NBR1] != NB) block[n][AMR_NBR1P] = block[block[n][AMR_NBR1]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_NBR1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 - 1, block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					block[n][AMR_NBR1P] = block[block[n][AMR_NBR1_3]][AMR_PARENT];
				}
			}
		}
		if (block[n][AMR_NBR2] >= 0){
			block[n][AMR_NBR2P] = block[block[n][AMR_NBR2]][AMR_PARENT];
		}
		if (block[n][AMR_NBR3] >= 0){
			if (block[n][AMR_NBR3] != NB){
				block[n][AMR_NBR3P] = block[block[n][AMR_NBR3]][AMR_PARENT];
			}
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					block[n][AMR_NBR3P] = block[block[n][AMR_NBR3_1]][AMR_PARENT];
				}
				else{ //Difference in REF_3
					block[n][AMR_NBR3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 + 1, block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
		}
		if (block[n][AMR_NBR4] >= 0){
			block[n][AMR_NBR4P] = block[block[n][AMR_NBR4]][AMR_PARENT];
		}
		if (block[n][AMR_NBR5] >= 0){
			block[n][AMR_NBR5P] = block[block[n][AMR_NBR5]][AMR_PARENT];
		}
		if (block[n][AMR_NBR6] >= 0){
			block[n][AMR_NBR6P] = block[block[n][AMR_NBR6]][AMR_PARENT];
		}
	}

	//Set corn parent
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));

		if (block[n][AMR_CORN1] >= 0){
			if (block[n][AMR_CORN1] != NB)block[n][AMR_CORN1P] = block[block[n][AMR_CORN1]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 - 1, block[n][AMR_COORD1] + 1, block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					if (block[n][AMR_CORN1_1] >= 0)block[n][AMR_CORN1P] = block[block[n][AMR_CORN1_1]][AMR_PARENT];
				}
			}
		}
		if (block[n][AMR_CORN2] >= 0){
			if (block[n][AMR_CORN2] != NB)block[n][AMR_CORN2P] = block[block[n][AMR_CORN2]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					if (block[n][AMR_CORN2_1] >= 0)block[n][AMR_CORN2P] = block[block[n][AMR_CORN2_1]][AMR_PARENT];
				}
				else{
					block[n][AMR_CORN2P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 + 1, block[n][AMR_COORD1] + 1, block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
		}
		if (block[n][AMR_CORN3] >= 0){
			if (block[n][AMR_CORN3] != NB)block[n][AMR_CORN3P] = block[block[n][AMR_CORN3]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					if (block[n][AMR_CORN3_1] >= 0)block[n][AMR_CORN3P] = block[block[n][AMR_CORN3_1]][AMR_PARENT];
				}
				else{
					block[n][AMR_CORN3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 + 1, block[n][AMR_COORD1] - 1, block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
		}
		if (block[n][AMR_CORN4] >= 0){
			if (block[n][AMR_CORN4] != NB)block[n][AMR_CORN4P] = block[block[n][AMR_CORN4]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN4P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 - 1, block[n][AMR_COORD1] - 1, block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					if (block[n][AMR_CORN4_1] >= 0)block[n][AMR_CORN4P] = block[block[n][AMR_CORN4_1]][AMR_PARENT];
				}
			}
		}
		if (block[n][AMR_CORN5] >= 0){
			block[n][AMR_CORN5P] = block[block[n][AMR_CORN5]][AMR_PARENT];
		}
		if (block[n][AMR_CORN6] >= 0){
			block[n][AMR_CORN6P] = block[block[n][AMR_CORN6]][AMR_PARENT];
		}
		if (block[n][AMR_CORN7] >= 0){
			block[n][AMR_CORN7P] = block[block[n][AMR_CORN7]][AMR_PARENT];
		}
		if (block[n][AMR_CORN8] >= 0){
			block[n][AMR_CORN8P] = block[block[n][AMR_CORN8]][AMR_PARENT];
		}
		if (block[n][AMR_CORN9] >= 0){
			if (block[n][AMR_CORN9] != NB)block[n][AMR_CORN9P] = block[block[n][AMR_CORN9]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN9P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 - 1, block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3) + REF_3);
				}
				else{
					if (block[n][AMR_CORN9_1] >= 0)block[n][AMR_CORN9P] = block[block[n][AMR_CORN9_1]][AMR_PARENT];
				}
			}
		}
		if (block[n][AMR_CORN10] >= 0){
			if (block[n][AMR_CORN10] != NB)block[n][AMR_CORN10P] = block[block[n][AMR_CORN10]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					if (block[n][AMR_CORN10_1] >= 0)block[n][AMR_CORN10P] = block[block[n][AMR_CORN10_1]][AMR_PARENT];
				}
				else{
					block[n][AMR_CORN10P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 + 1, block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3) + REF_3);
				}
			}
		}
		if (block[n][AMR_CORN11] >= 0){
			if (block[n][AMR_CORN11] != NB)block[n][AMR_CORN11P] = block[block[n][AMR_CORN11]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					if (block[n][AMR_CORN11_1] >= 0)block[n][AMR_CORN11P] = block[block[n][AMR_CORN11_1]][AMR_PARENT];
				}
				else{
					block[n][AMR_CORN11P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 + 1, block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3) - REF_3);
				}
			}
		}
		if (block[n][AMR_CORN12] >= 0){
			if (block[n][AMR_CORN12] != NB)block[n][AMR_CORN12P] = block[block[n][AMR_CORN12]][AMR_PARENT];
			else{
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN12P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0 - 1, block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3) - REF_3);
				}
				else{
					if (block[n][AMR_CORN12_1] >= 0)block[n][AMR_CORN12P] = block[block[n][AMR_CORN12_1]][AMR_PARENT];
				}
			}
		}
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
		block[n][AMR_GPU] = -1;

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

	int i,j,g, z, node, tt, fp, ip, y, rem, nr_timesteps, n_active_localsteps[NB];
	int i1, j1, z1, k, n, u, b, stride, count=0;
	int n_active_total_steps = 0, n_active_total_steps_t[10], steps_total_RM[NB];
	int NODE[NB], GPU[NB], temp;
	int n_active_total_t[10], (*n_ord_total_RM_t)[10], n_active_local_gpu[N_GPU], n_active_local_max,n_active_local_min;
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	int timelevel_cutoff = AMR_MAXTIMELEVEL;
	int numtasks_local = numtasks*N_GPU;
	int min_steps, max_steps, total_steps, count_gpu[N_GPU];
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	#if(DEREFINE_POLE && N_LEVELS_3D==1)
	rm_order1();
	#else
	rm_order2();
	#endif
	n_ord_total_RM_t=(int(*)[10])calloc(NB, sizeof(int[10]));
	if (numtasks_local > NB && rank == 0) fprintf(stderr, "Warning: numtasks_local is smaller than NB. Watch out for crashes! \n");
	do{
		count++;
		/*First make a z-order curve for each timelevel seperately, then load balance for timesteps. This is the best and most advanced method*/
		for (i = 0; i <= round(log(timelevel_cutoff) / log(2)); i++){
			n_active_total_t[i] = 0;
			n_active_total_steps_t[i] = 0;
		}
		int tl;
		//Order active blocks in an ordered array and keep track of the number of blocks and timesteps at each timelevel
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total_RM[n]][AMR_ACTIVE] == 1){
				tl = MY_MIN(round(log(block[n_ord_total_RM[n]][AMR_TIMELEVEL]) / log(2)), log(timelevel_cutoff) / log(2));
				n_ord_total_RM_t[n_active_total_t[tl]][tl] = n_ord_total_RM[n];
				n_active_total_steps_t[tl] += timelevel_cutoff / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL], timelevel_cutoff);
				n_active_total_t[tl]++;
			}
		}

		//Reset variables
		n_active_local_max = 0;
		for (g = 0; g < N_GPU;g++) n_active_local_gpu[g] = 0;
		n_active_local_min = 0;
		for (u = 0; u < MY_MIN(numtasks_local,NB); u++) n_active_localsteps[u] = 0;
		int increment = 0, n0, fillup_mode = 0;
		u = 0; //Initial node number
		int sw = 0;

		//Try to give each node the same number of lower timelevel blocks
		for (i = 0; i <= round(log(timelevel_cutoff) / log(2)); i++){
			//If there is not an even load from the previous timelevel, first correct for that
			if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] > n_active_localsteps[u % numtasks_local]){
				nr_timesteps = timelevel_cutoff / MY_MIN(block[n_ord_total_RM_t[0][i]][AMR_TIMELEVEL], timelevel_cutoff);
				increment = (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] - n_active_localsteps[u % numtasks_local]) / nr_timesteps;
				fillup_mode = 1;
			}
			if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] == n_active_localsteps[u % numtasks_local]){ //Otherwise just do nornmal load balancing
				rem = n_active_total_t[i] % (numtasks_local); //remainder number of blocks at given timelevel
				increment = (n_active_total_t[i] - rem) / (numtasks_local); //number of blocks/node at given timelevel
				fillup_mode = 0;
				sw = 1;
			}
			n = 0;
			while (n < n_active_total_t[i]){
				nr_timesteps = timelevel_cutoff / MY_MIN(block[n_ord_total_RM_t[n][i]][AMR_TIMELEVEL], timelevel_cutoff);
				if (fillup_mode == 1) increment = (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] - n_active_localsteps[u % numtasks_local]) / nr_timesteps;
				if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] == n_active_localsteps[u % numtasks_local]){
					rem = (n_active_total_t[i] - n) % (numtasks_local); //remainder number of blocks at given timelevel
					increment = (n_active_total_t[i] - n - rem) / (numtasks_local); //number of blocks/node at given timelevel
					fillup_mode = 0;
					sw = 1;
				}

				if (fillup_mode == 0 && ((n_active_total_t[i] - n) / (increment + 1)) == rem && (n_active_total_t[i] - n) % (increment + 1) == 0 && rem > 0){
					increment += 1;
					sw = 1;
				}
				increment = MY_MIN(increment, n_active_total_t[i] - n);
				for (j = 0; j < increment; j++){
					NODE[n_ord_total_RM_t[n + j][i]] = (u%numtasks_local);
					n_active_localsteps[u%numtasks_local] += nr_timesteps;
				}
				n += increment;
				if (n_active_localsteps[u%numtasks_local] == n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] || sw == 1){
					sw = 0;
					u = (u + 1) % numtasks_local;//Increase node number
				}
				if (increment == 0 && rank == 0) fprintf(stderr, "Load balance error \n");
			}
		}

		//Split up between GPUs on a single node
		for (n = 0; n < n_active_total; n++){
			temp = NODE[n_ord_total_RM[n]];
			NODE[n_ord_total_RM[n]] = temp / N_GPU;
			GPU[n_ord_total_RM[n]] = (temp - NODE[n_ord_total_RM[n]] * N_GPU);
			if (rank == NODE[n_ord_total_RM[n]])n_active_local_gpu[GPU[n_ord_total_RM[n]]]++;
			if (GPU[n_ord_total_RM[n]] >= 20) fprintf(stderr, "Catastrophic load balancing error 1 \n");
			if (NODE[n_ord_total_RM[n]] >= numtasks) fprintf(stderr, "Catastrophic load balancing error 2 \n");
		}
		for (g = 0; g < N_GPU; g++){
			n_active_local_max = MY_MAX(n_active_local_max, n_active_local_gpu[g]);
		}
		n_active_local_min = n_active_local_gpu[0];
		for (g = 1; g < N_GPU; g++){
			n_active_local_min = MY_MIN(n_active_local_min, n_active_local_gpu[g]);
		}
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);

		if ((n_active_local_max> MAX_BLOCKS || n_active_local_min < 1) && timelevel_cutoff >= 2) timelevel_cutoff /= 2;
	} while ((n_active_local_max> MAX_BLOCKS || n_active_local_min < 1) && count < round(log(AMR_MAXTIMELEVEL) / log(2)) + 1);

	if (rank == 0 && n_active_local_max > MAX_BLOCKS) fprintf(stderr, "Error in balance_load: Too many blocks refined, possible to get OpenCL or OOM errors! \n");
	if (rank == 0) fprintf(stderr, "Load balance started with cutoff timelevel %d! \n", timelevel_cutoff);
	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[n_ord_total_RM[i]]){
			if (block[n_ord_total_RM[i]][AMR_NODE] == rank){
				rc = MPI_Isend(&p[nl[n_ord_total_RM[i]]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[i]], (5 * NB_LOCAL + block[n_ord_total_RM[i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[i]]][0]);
				#if STAGGERED
				rc += MPI_Isend(&ps[nl[n_ord_total_RM[i]]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[i]], (6 * NB_LOCAL + block[n_ord_total_RM[i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[i]]][1]);
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
					rc = MPI_Irecv(&temp_p[n_ord_total_RM[i]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (5 * NB_LOCAL + block[n_ord_total_RM[i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp1[n_ord_total_RM[i]]);
					#if STAGGERED
					rc += MPI_Irecv(&temp_ps[n_ord_total_RM[i]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (6 * NB_LOCAL + block[n_ord_total_RM[i]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp2[n_ord_total_RM[i]]);
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
				free_arrays(n_ord_total_RM[i]);
				#if(GPU_ENABLED || GPU_DEBUG )
				GPU_finish(n_ord_total_RM[i], 0);
				#endif
			}
			block[n_ord_total_RM[i]][AMR_GPU] = -1;
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
			}
		}

		//Set to updated node
		block[n_ord_total_RM[n]][AMR_NODE] = NODE[n_ord_total_RM[n]];
	}

	//Now reloadbalance between the GPUs on a single node
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total_RM[n]][AMR_NODE] == rank){
			if (GPU[n_ord_total_RM[n]] != block[n_ord_total_RM[n]][AMR_GPU]){
				#if(GPU_ENABLED || GPU_DEBUG )
				set_arrays_GPU(n_ord_total_RM[n], GPU[n_ord_total_RM[n]]);
				GPU_write(n_ord_total_RM[n]);
				#endif
			}
		}
		block[n_ord_total_RM[n]][AMR_GPU] = GPU[n_ord_total_RM[n]];
	}

	activate_blocks();
	
	min_steps = 0; max_steps = 0; total_steps = 0;
	count_node[0]=0;
	for (g = 0; g < N_GPU; g++) count_gpu[g] = 0;
	for (n = 0; n < n_active; n++) count_gpu[block[n_ord[n]][AMR_GPU]] += AMR_MAXTIMELEVEL / block[n_ord[n]][AMR_TIMELEVEL];
	min_steps = count_gpu[0];
	for (g = 0; g < N_GPU; g++){
		max_steps = MY_MAX(max_steps, count_gpu[g]);
		min_steps = MY_MIN(min_steps, count_gpu[g]);
		total_steps += count_gpu[g];
	}
	MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &min_steps, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &max_steps, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &total_steps, 1, MPI_INT, MPI_SUM, mpi_cartcomm);

	if (rank == 0) fprintf(stderr, "Number of active blocks (total, min,max): %d %d %d \n", n_active_total, n_active_local_min, n_active_local_max);
	if (rank == 0) fprintf(stderr, "Number of active steps (total, min,max): %d %d %d \n", total_steps, min_steps, max_steps);

	bound_prim(p, 1);
	#if(GPU_ENABLED)
	GPU_boundprim(1);
	#endif
	free(n_ord_total_RM_t);
	if (rank == 0) fprintf(stderr, "Load balance finished! \n");
}

void balance_load_gpu(void){
}

/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
void activate_blocks(void){
	int n, i;
	n_active = 0;
	n_active_total = 0;

	for (n = 0; n < MY_MIN(numtasks * N_GPU, NB); n++) NODE_global[n] = 0;
	for (n = 0; n <= n_max; n++) block[n][AMR_REFINED] = 0;
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
            if (block[n][AMR_NODE] >= 0){
                block[n][AMR_NUMBER] = NODE_global[block[n][AMR_NODE]];
                NODE_global[block[n][AMR_NODE]]++;
            }
			n_active_total++;
			if (block[n][AMR_LEVEL] > 0) block[block[n][AMR_PARENT]][AMR_REFINED] = 1;
		}
	}
	#if(N_GPU>1)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_NODE] >= 0) NODE_global[block[n_ord_total[n]][AMR_NODE]*N_GPU + (block[n_ord_total[n]][AMR_GPU])]++;
	}
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
}

void block_average(int n, int n_child, int i1, int i2, int j1, int j2, int z1, int z2){
	int i, j, z, k, ic, jc, zc, i_1, i_2, j_1, j_2, z_1, z_2;
	struct of_geom geom;
	struct of_state q;
	int ref_1, ref_2, ref_3;

	ref_1 = block[n][AMR_LEVEL1] - block[n_child][AMR_LEVEL1];
	ref_2 = block[n][AMR_LEVEL2] - block[n_child][AMR_LEVEL2];
	ref_3 = block[n][AMR_LEVEL3] - block[n_child][AMR_LEVEL3];

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
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child], (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child], (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child], (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child] + ref_2, (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child], (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child], (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child] + ref_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child], (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child] + ref_2, (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child] + ref_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child] + ref_1, (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child], (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child] + ref_1, (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child] + ref_2, (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child])][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child] + ref_1, (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child], (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child] + ref_3)][k] +
							p[nl[n_child]][index_3D(n_child, (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child] + ref_1, (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child] + ref_2, (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child] + ref_3)][k]);
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
						i_1 = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child];
						i_2 = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child] + ref_1;
						j_1 = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child];
						j_2 = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child] + ref_2;
						z_1 = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child];
						z_2 = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child] + ref_3;

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
					ic = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child];
					k = 1;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + ref_3)][FACE1] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc + ref_3)][FACE1]);
				}
			}
		}
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2 + D2; j++){
				for (z = z1; z < z2; z++){
					ic = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child];
					k = 2;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + ref_3)][FACE2] +
						ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc + ref_3)][FACE2]);
				}
			}
		}
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2 + D3; z++){
					ic = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child];
					k = 3;
					ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] * 0.25*(
						ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc)][FACE3] +
						ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc + ref_2, zc)][FACE3]);
				}
			}
		}
		#endif
	}
}

void derefine(int n){
	int i,j,z,k, n_child;
	int ref_1, ref_2, ref_3;
	if (rank == 0) fprintf(stderr, "Derefining block %d %d %d %d \n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
	if (block[n][AMR_ACTIVE] != 0) fprintf(stderr, "Error: Trying to derefine active block %d! \n", n);

	block[n][AMR_ACTIVE] = 1;
	block[n][AMR_GPU] = block[block[n][AMR_CHILD1]][AMR_GPU];
	if (block[n][AMR_NODE] == rank){
		set_arrays(n);
		set_grid(n);

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD1]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD1]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD1]][AMR_LEVEL3];
		if (block[n][AMR_CHILD1] >= 0){
			n_child = block[n][AMR_CHILD1];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD1]);
			GPU_finish(block[n][AMR_CHILD1], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD2]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD2]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD2]][AMR_LEVEL3];
		if (block[n][AMR_CHILD2] >= 0 && ref_3 == 1){
			n_child = block[n][AMR_CHILD2];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD2]);
			GPU_finish(block[n][AMR_CHILD2], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD3]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD3]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD3]][AMR_LEVEL3];
		if (block[n][AMR_CHILD3] >= 0 && ref_2 == 1){
			n_child = block[n][AMR_CHILD3];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD3]);
			GPU_finish(block[n][AMR_CHILD3], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD4]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD4]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD4]][AMR_LEVEL3];
		if (block[n][AMR_CHILD4] >= 0 && ref_2 == 1 && ref_3 == 1){
			n_child = block[n][AMR_CHILD4];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD4]);
			GPU_finish(block[n][AMR_CHILD4], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD5]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD5]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD5]][AMR_LEVEL3];
		if (block[n][AMR_CHILD5] >= 0 && ref_1 == 1){
			n_child = block[n][AMR_CHILD5];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD5]);
			GPU_finish(block[n][AMR_CHILD5], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD6]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD6]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD6]][AMR_LEVEL3];
		if (block[n][AMR_CHILD6] >= 0 && ref_1 == 1 && ref_3 == 1){
			n_child = block[n][AMR_CHILD6];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD6]);
			GPU_finish(block[n][AMR_CHILD6], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD7]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD7]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD7]][AMR_LEVEL3];
		if (block[n][AMR_CHILD7] >= 0 && ref_1 == 1 && ref_2 == 1){
			n_child = block[n][AMR_CHILD7];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD7]);
			GPU_finish(block[n][AMR_CHILD7], 0);
			#endif
		}

		ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CHILD8]][AMR_LEVEL1];
		ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CHILD8]][AMR_LEVEL2];
		ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CHILD8]][AMR_LEVEL3];
		if (block[n][AMR_CHILD8] >= 0 && ref_1 == 1 && ref_2 == 1 && ref_3 == 1){
			n_child = block[n][AMR_CHILD8];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD8]);
			GPU_finish(block[n][AMR_CHILD8], 0);
			#endif
		}
	}
	int min_timelevel = AMR_MAXTIMELEVEL;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
		if (block[block[n][i]][AMR_TIMELEVEL] < min_timelevel) min_timelevel = block[block[n][i]][AMR_TIMELEVEL];
		block[block[n][i]][AMR_GPU] = -1;
	}
	block[n][AMR_TIMELEVEL] = MY_MIN(2 * min_timelevel, AMR_MAXTIMELEVEL);
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_ACTIVE] = 0;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_TIMELEVEL] = 1;

	#if(GPU_ENABLED || GPU_DEBUG )
	if (block[n][AMR_NODE] == rank){
		if (block[block[n][AMR_CHILD1]][AMR_GPU] == -1 && GPU_ENABLED) fprintf(stderr, "Only positive values allowed for device number! \n");
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
	int ref_1, ref_2, ref_3;

	ref_1 = block[n_child][AMR_LEVEL1] - block[n][AMR_LEVEL1];
	ref_2 = block[n_child][AMR_LEVEL2] - block[n][AMR_LEVEL2];
	ref_3 = block[n_child][AMR_LEVEL3] - block[n][AMR_LEVEL3];

	#pragma omp parallel private(i, j, z, i1, j1, z1, k, i_1, j_1, z_1, i_2, j_2, z_2,geom,q,U,U0,factor)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, BS_1 - 1, 0, BS_2 - 1, 0, BS_3 - 1) {
			i1 = (i - i % (1 + ref_1)) / (1 + ref_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * ref_1;
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * ref_2;
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * ref_3;
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 0){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] - 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 1){
				PLOOP{
					prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.25 * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
				}
			}
			//Enforce strict conservation of conservative quantitites during refinement
			/*if (i % (1 + ref_1) == ref_1 && j % (1 + ref_2) == ref_2 && z % (1 + ref_3) == ref_3){
				i1 = (i - i % (1 + ref_1)) / (1 + ref_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * ref_1;
				j1 = (j - j % (1 + ref_2)) / (1 + ref_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * ref_2;
				z1 = (z - z % (1 + ref_3)) / (1 + ref_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * ref_3;

				i_1 = N1_GPU_offset[n_child] + i - ref_1;
				i_2 = N1_GPU_offset[n_child] + i;
				j_1 = N2_GPU_offset[n_child] + j - ref_2;
				j_2 = N2_GPU_offset[n_child] + j;
				z_1 = N3_GPU_offset[n_child] + z - ref_3;
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
				if (ref_3 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]=Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)]);
				}
				if (ref_2 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_1, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_1)]);
				}
				if (ref_2==1 && ref_3 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_1, j_2, z_2)]);
				}
				if (ref_1 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_1)]);
				}
				if (ref_1 == 1 && ref_3==1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_1, z_2)]);
				}
				if (ref_1 == 1 && ref_2 == 1){
					PLOOP dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_2, z_1, CENT, &geom);
					pflag[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)] = Utoprim_2d(dq[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[nl[n_child]][index_3D(n_child, i_2, j_2, z_1)]);
				}
				if (ref_1 == 1 && ref_2 == 1 && ref_3 == 1){
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
	int ref_1, ref_2, ref_3;
	double *pointer1, *pointer2, *pointer3, *pointer4, *pointer5, *pointer6;

	ref_1 = block[n_child][AMR_LEVEL1] - block[n][AMR_LEVEL1];
	ref_2 = block[n_child][AMR_LEVEL2] - block[n][AMR_LEVEL2];
	ref_3 = block[n_child][AMR_LEVEL3] - block[n][AMR_LEVEL3];

	//Use divergence free prolongation to handle boundaries
	n_rec1 = -1;
	n_rec2 = -1;
	n_rec3 = -1;
	n_rec4 = -1;
	n_rec5 = -1;
	n_rec6 = -1;

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){ pointer4 = receive4_5[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){ pointer4 = receive4_6[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){ pointer4 = receive4_7[nl[n]]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_1]>=0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1){ pointer4 = receive4_8[nl[n]]; n_rec4 = 1; }

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){ pointer2 = receive2_1[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){ pointer2 = receive2_2[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){ pointer2 = receive2_3[nl[n]]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_5]>=0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1){ pointer2 = receive2_4[nl[n]]; n_rec2 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_1[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_2[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_5[nl[n]]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_3]>=0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_6[nl[n]]; n_rec3 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_3[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_4[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_7[nl[n]]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_1]>=0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_8[nl[n]]; n_rec1 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){ pointer5 = receive5_1[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){ pointer5 = receive5_3[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){ pointer5 = receive5_5[nl[n]]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_2]>=0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1){ pointer5 = receive5_7[nl[n]]; n_rec5 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){ pointer6 = receive6_2[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){ pointer6 = receive6_4[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){ pointer6 = receive6_6[nl[n]]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_1]>=0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1){ pointer6 = receive6_8[nl[n]]; n_rec6 = 1; }
	
	isize = BS_1;
	jsize = BS_2;
	zsize = BS_3;
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, ind0, ind2,b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8,b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8,set_1, set_2,set_3,set_4,set_5,set_6)
	{
	#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, BS_1 - 1 + D1, 0, BS_2 - 1 + D2, 0, BS_3 - 1 + D3) {
			//indices in coarse grid depending on offset (input parameter)
			i1 = (i - i % (1 + ref_1)) / (1 + ref_1);
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2);
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3);

			//index of child
			ind0 = index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);
			ind2 = index_2D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);

			//face centered magnetic field components of neighbouring cells that are refined allready
			b1_1 = b1_2 = b1_3 = b1_4 = b1_5 = b1_6 = b1_7 = b1_8 = 0.;
			b2_1 = b2_2 = b2_3 = b2_4 = b2_5 = b2_6 = b2_7 = b2_8 = 0.;
			b3_1 = b3_2 = b3_3 = b3_4 = b3_5 = b3_6 = b3_7 = b3_8 = 0.;
			set_1 = set_2 = set_3 = set_4 = set_5 = set_6 = -1;
			if ((i == 0 || i == ref_1) && offset_1 == 0 && n_rec2 >= 0){
				b1_1 = pointer2[j1*(1 + ref_2)*zsize + z1*(1 + ref_3)];
				b1_2 = pointer2[j1*(1 + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				b1_3 = pointer2[(j1*(1 + ref_2) + ref_2)*zsize + z1*(1 + ref_3)];
				b1_4 = pointer2[(j1*(1 + ref_2) + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				set_2 = 1;
			}
			else set_2 = -1;
			if ((i == BS_1 || i == BS_1 - ref_1 || i == BS_1 - (1 + ref_1)) && (offset_1 == 1 || ref_1 == 0) && n_rec4 >= 0){
				b1_5 = pointer4[j1*(1 + ref_2)*zsize + z1*(1 + ref_3)];
				b1_6 = pointer4[j1*(1 + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				b1_7 = pointer4[(j1*(1 + ref_2) + ref_2)*zsize + z1*(1 + ref_3)];
				b1_8 = pointer4[(j1*(1 + ref_2) + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				set_4 = 1;
			}
			else set_4 = -1;

			if ((j == 0 || j == ref_2) && offset_2 == 0 && n_rec3 >= 0){
				b2_1 = pointer3[i1*(1 + ref_1)*zsize + z1*(1 + ref_3)];
				b2_2 = pointer3[i1*(1 + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				b2_5 = pointer3[(i1*(1 + ref_1) + ref_1)*zsize + z1*(1 + ref_3)];
				b2_6 = pointer3[(i1*(1 + ref_1) + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				set_3 = 1;
			}
			else set_3 = -1;
			if ((j == BS_2 || j == BS_2 - ref_2 || j == BS_2 - (1 + ref_2)) && (offset_2 == 1 || ref_2 == 0) && n_rec1 >= 0){
				b2_3 = pointer1[i1*(1 + ref_1)*zsize + z1*(1 + ref_3)];
				b2_4 = pointer1[i1*(1 + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				b2_7 = pointer1[(i1*(1 + ref_1) + ref_1)*zsize + z1*(1 + ref_3)];
				b2_8 = pointer1[(i1*(1 + ref_1) + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				set_1 = 1;
			}
			else set_1 = -1;

			if ((z == 0 || z == ref_3) && offset_3 == 0 && n_rec5 >= 0){
				b3_1 = pointer5[i1*(1 + ref_1)*jsize + j1*(1 + ref_2)];
				b3_3 = pointer5[i1*(1 + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				b3_5 = pointer5[(i1*(1 + ref_1) + ref_1)*jsize + j1*(1 + ref_2)];
				b3_7 = pointer5[(i1*(1 + ref_1) + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				set_5 = 1;
			}
			else set_5 = -1;
			if ((z == BS_3 || z == BS_3 - ref_3 || z == BS_3 - (1 + ref_3)) && (offset_3 == 1 || ref_3 == 0) && n_rec6 >= 0){
				b3_2 = pointer6[i1*(1 + ref_1)*jsize + j1*(1 + ref_2)];
				b3_4 = pointer6[i1*(1 + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				b3_6 = pointer6[(i1*(1 + ref_1) + ref_1)*jsize + j1*(1 + ref_2)];
				b3_8 = pointer6[(i1*(1 + ref_1) + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				set_6 = 1;
			}
			else set_6 = -1;

			i1 = (i - i % (1 + ref_1)) / (1 + ref_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * ref_1 - (i == BS_1 && (offset_1 == 1 || ref_1 == 0));
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * ref_2 - (j == BS_2 && (offset_2 == 1 || ref_2 == 0));
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * ref_3 - (z == BS_3 && (offset_3 == 1 || ref_3 == 0));

			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || ref_1 == 0)), -0.25*ref_2, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*ref_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || ref_2 == 0)), -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*ref_1, -0.25*ref_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || ref_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}

			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || ref_1 == 0)), -0.25*ref_2, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*ref_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || ref_2 == 0)), 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*ref_1, -0.25*ref_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || ref_1 == 0)), 0.25*ref_2, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*ref_1, 0.0, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*ref_1, 0.25*ref_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || ref_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 0 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == BS_1 && (offset_1 == 1 || ref_1 == 0)), 0.25*ref_2, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*ref_1, 0.0, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*ref_1, 0.25*ref_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*ref_2, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*ref_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || ref_2 == 0)), -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*ref_1, -0.25*ref_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || ref_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 0 && z % (1 + ref_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*ref_2, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*ref_1, -0.5 + (double)(j == BS_2 && (offset_2 == 1 || ref_2 == 0)), 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*ref_1, -0.25*ref_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 0){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*ref_2, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*ref_1, 0.0, -0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*ref_1, 0.25*ref_2, -0.5 + (double)(z == BS_3 && (offset_3 == 1 || ref_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + ref_1) == 1 && j % (1 + ref_2) == 1 && z % (1 + ref_3) == 1){
				pb[nl[n_child]][ind0][1] =
					1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*ref_2, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][2] =
					1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*ref_1, 0.0, 0.25*ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[nl[n_child]][ind0][3] =
					1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*ref_1, 0.25*ref_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
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

	gpu = 0;
	rc = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	for (n1 = 0; n1 < n_active; n1++)Bp_send1(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec1(n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_send2(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec2(n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_send3(psh, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++)Bp_rec3(n_ord[n1]);
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB_AMR \n");
}

int refine(int n){
	int i, j, z, k, n_child, i1, j1, z1, n1, gpu_local;
	int ref_1, ref_2, ref_3;
	//MPI_Barrier(mpi_cartcomm);
	if (!check_nesting(n) || NODE_global[block[n][AMR_NODE]*N_GPU + block[n][AMR_GPU]] > MAX_BLOCKS){
		if (rank == 0) fprintf(stderr, "Failed to refine block %d %d %d %d due to memory size on node %d!\n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3], block[n][AMR_NODE]);
		return 0; //First make sure nesting criteria are satisfied
	}
	else{
		if (rank == 0) fprintf(stderr, "Refining block %d %d %d %d on node %d\n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3], block[n][AMR_NODE]);
		NODE_global[block[n][AMR_NODE] * N_GPU + block[n][AMR_GPU]] += (1 + REF_1)*(1 + REF_2)*(1 + REF_3) - 1;
	}

	if (rank == 0) if (block[n][AMR_ACTIVE] != 1) fprintf(stderr,"Error trying to refine non-active block %d \n", n);

	gpu_local = block[n][AMR_GPU];
	if (block[n][AMR_NODE] == rank){
		//Calculate gradients, store in flux array F1, F2, F3
		#pragma omp parallel private(i, j, z,k)
		{
			#pragma omp for collapse(2) schedule(dynamic)
			ZSLOOP3D(-D1, BS_1 - 1 + D1, -D2, BS_2 - 1 + D2, -D3, BS_3 - 1 + D3) {
				PLOOP{
					F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n] - 1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n] + 1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
					F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] - 1, z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + 1, z + N3_GPU_offset[n])][k]);
					#if(N3>1)
					F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] - 1)][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + 1)][k]);
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

		ref_3 = block[block[n][AMR_CHILD2]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD2] >= 0 && ref_3 == 1){
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

		ref_2 = block[block[n][AMR_CHILD3]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		if (block[n][AMR_CHILD3] >= 0 && ref_2 == 1){
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

		ref_2 = block[block[n][AMR_CHILD4]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD4]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD4] >= 0 && ref_2 == 1 && ref_3 == 1){
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

		ref_1 = block[block[n][AMR_CHILD5]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_CHILD5] >= 0 && ref_1 == 1){
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

		ref_1 = block[block[n][AMR_CHILD6]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_3 = block[block[n][AMR_CHILD6]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD6] >= 0 && ref_1 == 1 && ref_3 == 1){
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

		ref_1 = block[block[n][AMR_CHILD7]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD7]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		if (block[n][AMR_CHILD7] >= 0 && ref_1 == 1 && ref_2 == 1){
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

		ref_1 = block[block[n][AMR_CHILD8]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD8]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD8]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD8] >= 0 && ref_1 == 1 && ref_2 == 1 && ref_3 == 1){
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
	free_arrays(n);
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_finish(n, 0);
	#endif
}
	//MPI_Barrier(mpi_cartcomm);
	//Take note that block becomes refined
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
		block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL];
		if (block[n][AMR_TIMELEVEL] >= 2)block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL] / 2;
		else reduce_timestep = 1;
		block[block[n][i]][AMR_NODE] = block[n][AMR_NODE];
		block[block[n][i]][AMR_GPU] = gpu_local;
		block[block[n][i]][AMR_ACTIVE] = 1;
	}
	block[n][AMR_GPU]= -1;
	block[n][AMR_ACTIVE] = 0;
	block[n][AMR_TIMELEVEL] = 1;
	return 1;
}

void post_refine(void){
	//Allocate memory for all active blocks
	activate_blocks();
	set_corners();

	//Set boundary conditions
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG)
	MPI_Barrier(mpi_cartcomm);
	GPU_boundprim(1);
	MPI_Barrier(mpi_cartcomm);
	#endif
}

//Checks if neighbouring blocks are sufficiently refined so that no double jumps in refinement level are created
int check_nesting(int n){
	int i,z;
	int flag = 1;
	for (i = AMR_NBR1P; i <= AMR_CORN12P; i++){
		if (block[n][i] >= 0 && block[block[n][i]][AMR_ACTIVE] == 1){
			if (!refine(block[n][i])) flag = 0;
		}
	}	

	//Refine around pole
	if (block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1){
		if (rank==0) fprintf(stderr, "Warning refining around pole. This is not well tested, watch out for errors! \n");
		for (z = 0; z < NB_2*pow(1 + REF_3, block[n][AMR_LEVEL3]); z++) block[AMR_coord_linear2(block[n][AMR_LEVEL], block[n][AMR_COORD2]/pow(1 + REF_2, block[n][AMR_LEVEL2]), block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3])][AMR_TAG] = 1;
		flag = 0;
	}

	return flag;
}

#if WHICHPROBLEM==DISRUPTION_PROBLEM
#define REFINEMENT_CUTOFF 0.0000001
#else
#define REFINEMENT_CUTOFF 0.2 //in this case density in code units, used for H/R=0.03 disk
#endif

//Refine on basis of some criteria ref_val (not necessary to use rho though, can also be something different)
void check_refcrit(void){
	int n, task, i,j,z,k, l, level, number, ref_3;
	int node, n_send, gpu_choice, gpu_counter;
	double  rho_rec;
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	if (max_levels == 0) max_levels = N_LEVELS_3D;
	int tag, count, begin1, end1;
	int one_block_refined = 0, one_block_derefined=0;
	
	if(rank==0) fprintf(stderr,"Starting refinement! \n");

	//First close dump files in progress
	close_dump();
	close_rdump();
	close_gdump();
	MPI_Barrier(mpi_cartcomm);

	begin1 = time(NULL);
	count = 0;
	do{
		count++;
		tag = 0;

		/*Only allow refinement for one block per node per step*/
		for (i = 0; i < MY_MIN(numtasks * N_GPU, NB); i++){
			NODE_global[i] = 0;
		}

		//Count the number of blocks per node and reset tag
		for (n = 0; n < n_active_total; n++){
			block[n_ord_total[n]][AMR_TAG] = 0;
			NODE_global[block[n_ord_total[n]][AMR_NODE]]++;
		}

		/*First make sure all nodes have the same ref_val*/
		synch_refcrit();

		//Tag for refinement
		for (n = 0; n < n_active_total; n++){
			if ((ref_val[n_ord_total[n]] > REFINEMENT_CUTOFF || block[n_ord_total[n]][AMR_TAG] == 1) && block[n_ord_total[n]][AMR_LEVEL1] < max_levels - 1 && block[n_ord_total[n]][AMR_ACTIVE] == 1){ //If satisfy refinement criterion and smaller than maximum levels
				block[n_ord_total[n]][AMR_TAG] = 1;

				//Refine one level less near black hole
				level = block[n_ord_total[n]][AMR_LEVEL1];
				#if(!REFINE_JET)
				if (block[n_ord_total[n]][AMR_LEVEL1] == max_levels - 2 && block[n_ord_total[n]][AMR_COORD1] == 0){
					block[n_ord_total[n]][AMR_TAG] = 0;
				}
				#else
				if (block[n_ord_total[n]][AMR_COORD1] <= 0 && block[n_ord_total[n]][AMR_LEVEL1]==0) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 2 && block[n_ord_total[n]][AMR_LEVEL1] == 1) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 6 && block[n_ord_total[n]][AMR_LEVEL1] == 2) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 14 && block[n_ord_total[n]][AMR_LEVEL1] == 3) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 30 && block[n_ord_total[n]][AMR_LEVEL1] == 4) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 62 && block[n_ord_total[n]][AMR_LEVEL1] == 5) block[n_ord_total[n]][AMR_TAG] = 0;
				else if (block[n_ord_total[n]][AMR_COORD1] <= 126 && block[n_ord_total[n]][AMR_LEVEL1] == 6) block[n_ord_total[n]][AMR_TAG] = 0;
				#endif

				//Do not refine around both poles
				/*number = 0;
				if (block[n_ord_total[n]][AMR_LEVEL2] == 0) number = 0;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 1) number = 2;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 2) number = 6;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 3) number = 14;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 4) number = 30;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 5) number = 62;
				else if (block[n_ord_total[n]][AMR_LEVEL2] == 6) number = 126;
				if (REF_2 == 0) number = level;
				if ((block[n_ord_total[n]][AMR_COORD2] <= number || block[n_ord_total[n]][AMR_COORD2] >= NB_2*pow(1 + REF_2, level) - number - 1)){
					block[n_ord_total[n]][AMR_TAG] = 0;
				}*/

				if (block[n_ord_total[n]][AMR_TAG] == 1){
					if (one_block_refined == 0){
						pre_refine();
						one_block_refined = 1;
					}

					//Check if refinement indeed happened
					if (!refine(n_ord_total[n])){
						tag = 1;
					}
				}
			}
		}

		if(one_block_refined==1) post_refine();
		if (tag != 0 && n_active_total<numtasks*MAX_BLOCKS){
			balance_load();
			#if(GPU_ENABLED)
			balance_load_gpu();
			#endif
			pre_refine();
		}
	} while (tag != 0 && n_active_total<numtasks*MAX_BLOCKS && count<10);

	if (tag == 1){
		if(rank==0) fprintf(stderr, "Maximum number of blocks exceeded. Please select more nodes or adjust refinement criterion! \n");
		exit(0);
	}

	//First make sure all nodes have the same ref_val
	if (one_block_refined == 1) synch_refcrit();

	one_block_derefined = 0;
	count = 0;
	gpu_counter = 0;
	do{
		count++;
		tag = 0;
		for (n = 0; n < n_active_total; n++){
			//derefine
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[n_ord_total[n]][AMR_LEVEL1] > 0){
				block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = -1; //Tag for derefinement

				for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
					if (block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_REFINED] == 1)block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1; //If one of the children of the parent block is refined
					if (ref_val[block[block[n_ord_total[n]][AMR_PARENT]][i]] > 0.5*REFINEMENT_CUTOFF) block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1; //Except if one of the children does satisfy the refinement criterion
				}
			}
		}

		for (l = 0; l < max_levels; l++){
			for (n = 0; n < n_active_total; n++){
				//do not derefine if required for proper nesting
				for (i = AMR_NBR1; i <= AMR_CORN12; i++){
					if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[n_ord_total[n]][i] >= 0 && block[block[n_ord_total[n]][i]][AMR_TAG] >= 1){
						block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 2;
					}
				}
			}
		}

		//Do not derefine block around pole
		for (n = 0; n < n_active_total; n++){
			if (block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] > 0 && (block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == 0 || block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == NB_2*pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) - 1)){
				for (z = 0; z < NB_2*pow(1 + REF_3, block[n][AMR_LEVEL3]); z++){
					block[AMR_coord_linear2(block[n_ord_total[n]][AMR_LEVEL], block[n][AMR_COORD2]
					/ pow(1 + REF_2, block[n_ord_total[n]][AMR_LEVEL2]), block[n_ord_total[n]][AMR_COORD1], block[n_ord_total[n]][AMR_COORD2], block[n_ord_total[n]][AMR_COORD3])][AMR_TAG] = 2;
				}
			}
		}

		//Detag if load balancing required as intermediate step
		for (n = 0; n < n_active_total; n++){
			node = block[n_ord_total[n]][AMR_NODE];
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				if (NODE_global[node*N_GPU + block[n_ord_total[n]][AMR_GPU]] < MAX_BLOCKS + (1 + REF_3)*(1 + REF_2)*(1 + REF_1) - 1){
					ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
					for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
						n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
						if (block[n_send][AMR_NODE] != node){
							NODE_global[node*N_GPU + block[n_ord_total[n]][AMR_GPU]]++;
						}
					}
				}
				else{
					block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == 0;
					tag = 1;
				}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Send blocks
		for (n = 0; n < n_active_total; n++){
			node = block[n_ord_total[n]][AMR_NODE];
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				//Send block using non-blocking send
				ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
					n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
					if (block[n_send][AMR_NODE] != node){
						rc = 0;
						if (block[n_send][AMR_NODE] == rank){
							rc += MPI_Isend(&p[nl[n_send]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (3 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][598]);
							#if STAGGERED
							rc += MPI_Isend(&ps[nl[n_send]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (4 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][597]);
							#endif
						}
						if (rc != 0)fprintf(stderr, "Error in MPI in derefine \n");
					}
				}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Receive blocks
		for (n = 0; n < n_active_total; n++){
			node = block[n_ord_total[n]][AMR_NODE];
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				//Send block using non-blocking send
				ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
					n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
					if (block[n_send][AMR_NODE] != node){
						rc = 0;
						if (node == rank){
							//Allocate memory for active blocks on node
							temp_p[n_send] = (double(*)[NPR])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NPR]));
							temp_ps[n_send] = (double(*)[NDIM])calloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G), sizeof(double[NDIM]));
							if (block[n_send][AMR_NODE] >= 0){
								rc += MPI_Irecv(&temp_p[n_send][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (3 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp1[n_send]);
								#if STAGGERED
								rc += MPI_Irecv(&temp_ps[n_send][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (4 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp2[n_send]);
								#endif
							}
						}
						if (rc != 0)fprintf(stderr, "Error in MPI in derefine \n");
					}
				}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Clean up on sending side
		for (n = 0; n < n_active_total; n++){
			node = block[n_ord_total[n]][AMR_NODE];
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
					//Then use MPI_wait to clean up data that has been sent
					n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
					if (block[n_send][AMR_NODE] != node){
						if (block[n_send][AMR_NODE] == rank){
							MPI_Wait(&boundreqs[nl[n_send]][598], &Statbound[nl[n_send]][0]);
							#if STAGGERED
							MPI_Wait(&boundreqs[nl[n_send]][597], &Statbound[nl[n_send]][1]);
							#endif
							free_arrays(n_send);
							#if(GPU_ENABLED || GPU_DEBUG )
							GPU_finish(n_send, 0);
							#endif
						}
					}
				}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Allocate arrays on receiving side
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				node = block[n_ord_total[n]][AMR_NODE];
				ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
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
								for (k = 0; k < NDIM; k++) ps[nl[n_send]][index_3D(n_send, i, j, z)][k] = temp_ps[n_send][index_3D(n_send, i, j, z)][k];
							}
							free(temp_p[n_send]);
							free(temp_ps[n_send]);
							#if(GPU_ENABLED || GPU_DEBUG )
							/*if (mem_spot_gpu[nl[n_send]] != -1) gpu_choice = mem_spot_gpu[nl[n_send]];
							else {
								gpu_choice = gpu_counter%N_GPU;
								gpu_counter++;
							}
							set_arrays_GPU(n_send, gpu_choice);
							GPU_write(n_send);*/
							#endif
						}
					}
				}
			}
		}

		//Derefine if tagged for derefinement and not part of nesting
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]){
				node = block[n_ord_total[n]][AMR_NODE];
				ref_3 = block[n_ord_total[n]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3];
				for (i = AMR_CHILD1; i <= AMR_CHILD8; i += (2 - ref_3)){
					n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
					block[n_send][AMR_NODE] = node;
				}

				block[block[n_ord_total[n]][AMR_PARENT]][AMR_NODE] = node;

				//Then derefine and set corresponding tag and timelevel
				one_block_derefined = 1;
				derefine(block[n_ord_total[n]][AMR_PARENT]);
				block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 0;
			}
		}

		if (one_block_derefined == 1)post_refine();

		balance_load();
		#if(GPU_ENABLED)
		balance_load_gpu();
		#endif
	}while (tag != 0 && count<10);

	if (tag == 1){
		if (rank == 0) fprintf(stderr, "Derefinement ran out of memory! \n");
		exit(0);
	}

	//Decrease the timestep if required
	if (reduce_timestep == 1) dt /= 2.;
	reduce_timestep = 0;

	//Set timelevel communicator
	set_communicator();
	
	//Start very conservatively
	dt /= 2.;
	for (n = 0; n < n_active_total; n++){
		block[n_ord_total[n]][AMR_TIMELEVEL] = 1;
	}

	MPI_Barrier(mpi_cartcomm);
	end1 = time(NULL);
	if (rank == 0) fprintf(stderr, "Runtime load balance: %f \n", (double)(end1 - begin1));
}

//This function derefines in phi near the pole
int derefine_pole(void){
	int i, j, z, l, ni, nj, nz, u;
	//if (REF_3 != 1 || REF_1 == 1 || REF_2 == 1){
		//if(rank==0)fprintf(stderr, "Error! Derefinement near the pole works only for REF_1=0, REF_2=0, REF_3=1 \n");
		//exit(20);
		//return -1;
	//}
	if(rank==0)fprintf(stderr, "Derefining in phy by %d levels! \n", N_LEVELS_1D);

	if (NB_2 != 6 && NB_2 != 12 && NB_2 != 24 && NB_2 != 48 && NB_2 != 96){
		if (rank == 0)fprintf(stderr, "For derefinement near the pole chose NB_2 6, 12, 24, 48, 96 for 1, 2, 3, 4 levels of derefinement near the pole! \n");
		exit(20);
		return -1;
	}
	if (calc_mem(NB_1*NB_2*NB_3*pow(2., N_LEVELS_1D - 1)) > ((double)numtasks*(double)(numdevices)* 4. * (pow(10., 9.))) && rank == 1) fprintf(stderr, "You are exceeding the maximum memory size of 4 GB per GPU by refining too many blocks! Code will probably segfault, choose a bigger cluster \n");
	for (l = 0; l < N_LEVELS_1D; l++){
		pre_refine();
		ni = NB_1*pow(1 + REF_1, 0);
		nj = NB_2*pow(1 + REF_2, 0);
		nz = NB_3*pow(1 + REF_3, l);
		for (i = 0; i < ni; i++)for (j = pow(2, l); j < nj - (pow(2, l)); j++)for (z = 0; z < nz; z++){
			if ((double)pow(2, l) < 0.25*NB_2){
				if (!refine(AMR_coord_linear2(l,j, i, j, z))){
					if (rank == 0) fprintf(stderr, "Maximum number of blocks exceeded. Please select more nodes or adjust refinement criterion! \n");
					exit(0);
				}
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
void rm_order1(void){
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

//Set row major order in case of no derfinement near pole
void rm_order2(void){
	int n, l, i, j, z, number;
	int counter = 0;
	int counter2 = 0;
	for (n = 0; n <= n_max; n++){
		AMR_coord_cart_RM(n, &l, &i, &j, &z); //transform to cartesian grid coordinates
		number = AMR_coord_linear2(l,j/pow(1+REF_2,block[n][AMR_LEVEL2]), i, j, z); //transform to normal lineair ordering
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

//Calculate refinement criterion
double calc_refcrit(int n){
	int i, j, z;
	double ref_val = 0.0, enth, bsq, r, th, phi, X[NDIM];
	struct of_state q;
	struct of_geom geom;
	#if(REFINE_JET)
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (r > 50.0){
				get_geometry(n, i, j, z, CENT, &geom);
				get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);
				bsq = bsq_calc(p[nl[n]][index_3D(n, i, j, z)], &geom);
				if (log(q.ucon[0]) / log(10.0) > 0.5 || log(bsq / p[nl[n]][index_3D(n, i, j, z)][RHO]) / log(10.0) > 1.0 || log(p[nl[n]][index_3D(n, i, j, z)][UU] / p[nl[n]][index_3D(n, i, j, z)][RHO]) / log(10.0) > -0.2) ref_val = 100.0;
			}
		}
	}
	#elif(WHICHPROBLEM==DISRUPTION_PROBLEM)
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			enth=1.0+p[nl[n]][index_3D(n, i, j, z)][UU]*gam/p[nl[n]][index_3D(n, i, j, z)][RHO];
			if (p[nl[n]][index_3D(n, i, j, z)][RHO]*fabs(enth) > ref_val) ref_val = p[nl[n]][index_3D(n, i, j, z)][RHO]*enth;
		}
	}
	#else
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			if (p[nl[n]][index_3D(n, i, j, z)][RHO] > ref_val) ref_val = p[nl[n]][index_3D(n, i, j, z)][RHO];
		}
	}
	#endif
	return ref_val;
}

//Send refinement criterion across cluster
void synch_refcrit(void){
	int n, task;
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
			ref_val[n_ord_total[n]] = calc_refcrit(n_ord_total[n]);
			for (task = 0; task < numtasks; task++){
				if (rank != task){
					rc = MPI_Isend(&ref_val[n_ord_total[n]], 1, MPI_DOUBLE, task, (17 * NB_LOCAL + block[n_ord_total[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
					MPI_Request_free(&req[0]);
				}
			}
		}
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			rc = MPI_Irecv(&ref_val[n_ord_total[n]], 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], (17 * NB_LOCAL + block[n_ord_total[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
		}
	}
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
		}
	}
}

//Calculates RAM requirements in bytes (conservatively)
double calc_mem(int n_blocks){
	return n_blocks*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * 1000;
}
