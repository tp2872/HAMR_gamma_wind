#include "include.h"
#include "decs_MPI.h"

int AMR_coord_linear_RM(int level, int i, int j, int z);
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z);
void rm_order2(void);
void AMR_set_coord(void);

//For REFINE_GRB
#define BASE_LEVELS (0)//Number of refinement levels before focusing on jet
#define HIGH_CUTOFF (1E-1) //Define the cutoff for entropy in jet (HIGH_CUTOFF) and the equivalent for the coccoon (LOW_CUTOFF)//
#define LOW_CUTOFF (1E-3)
#define t_ff (pow(R_BONDI,1.5)) //Define the free-fall time for initial gap to close//
#define BLOCKS_PER_THETA_JET (1) //Minimum number of blocks per *half* opening angle of the jet
#define BLOCKS_PER_THETA_COCCOON (1) //Minimum number of blocks per *half* opening angle of the coccoon

//Define for REFINE_THIN the target number of cells per scaleheight
#define CELLS_PER_SCALEHEIGHT (15.0)

void test_AMR(void){
}

int AMR_coord_linear(int l, int i, int j, int z){
	fprintf(stderr, "Error: AMR_coord_linear is not used anymore! \n");
	exit(0);
	return -1;
}

int AMR_coord_linear2(int l, int b2, int i, int j, int z){
	int index, n, offset, L_1DMAX;
	int min_i, min_j, min_z;
	int max_i, max_j, max_z;
	int n_levels_tot;

	#if(GRID_CARTESIAN)
	if (l >= N_LEVELS_3D) {
		//Calculate minimum index
		min_i = NB_1 * pow(1 + REF_1, l) / 2 - NB_1 * pow(1 + REF_1, N_LEVELS_3D - 1) / 2;
		min_j = NB_2 * pow(1 + REF_2, l) / 2 - NB_2 * pow(1 + REF_2, N_LEVELS_3D - 1) / 2;
		min_z = NB_3 * pow(1 + REF_3, l) / 2 - NB_3 * pow(1 + REF_3, N_LEVELS_3D - 1) / 2;

		//Calculate maximum index
		max_i = NB_1 * pow(1 + REF_1, l) / 2 + NB_1 * pow(1 + REF_1, N_LEVELS_3D - 1) / 2;
		max_j = NB_2 * pow(1 + REF_2, l) / 2 + NB_2 * pow(1 + REF_2, N_LEVELS_3D - 1) / 2;
		max_z = NB_3 * pow(1 + REF_3, l) / 2 + NB_3 * pow(1 + REF_3, N_LEVELS_3D - 1) / 2;
	}
	else {
		min_i = min_j = min_z = 0;
		max_i = NB_1 * pow(1 + REF_1, l);
		max_j = NB_2 * pow(1 + REF_2, l);
		max_z = NB_3 * pow(1 + REF_3, l);
	}

	//Calculate maximum number of refinement levels 
	n_levels_tot = calc_nlevels_tot(l, i, j, z);

	//Either reject coordinates (n=-1) or set to special values (n=NB) at e.g. boundaries where only a parent bound exists
	if (l < 0 || l >= n_levels_tot || i < min_i || j < min_j || z < min_z || i >= max_i || j >= max_j || z >= max_j) {
		if (l < 0 || l >= n_levels_tot || i < 0 || j < 0 || z < 0 || i >= NB_1 * pow(1 + REF_1, l) || j >= NB_2 * pow(1 + REF_2, l) || z >= NB_3 * pow(1 + REF_3, l)) {
			n = -1;
		}
		else { 
			n = NB;
		}
	}
	else {
		if (l >= N_LEVELS_3D) {
			index = (int)((i - min_i) * NB_3 * (int)pow(1 + REF_3, N_LEVELS_3D - 1) * NB_2 * pow(1 + REF_2, N_LEVELS_3D - 1) + (j - min_j) * NB_3 * pow(1 + REF_3, N_LEVELS_3D - 1) + (z - min_z));
		}
		else {
			index = (int)((i - min_i) * NB_3 * (int)pow(1 + REF_3, l) * NB_2 * pow(1 + REF_2, l) + (j - min_j) * NB_3 * pow(1 + REF_3, l) + (z - min_z));
		}
		n = lin_coord[l][index];
		if (block[n][AMR_LEVEL] != l || block[n][AMR_COORD1] != i || block[n][AMR_COORD2] != j || block[n][AMR_COORD3] != z) {
			fprintf(stderr, "Could not find the right linear coordinate, input incorrect! \n");
			fprintf(stderr, "Incorrect values are l: %d j0: %d i: %d j: %d z: %d \n", l, b2, i, j, z);
			exit(0);
		}
	}
	#else
	if (b2 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)b2 + 1.0) / log(2.)), N_LEVELS_1D);
	else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - b2) + 1.0) / log(2.)), N_LEVELS_1D);

	if (l<0 || b2<0 || i<0 || j<0 || z<0 || i >= NB_1*pow(1 + (l>L_1DMAX)*REF_1, l - L_1DMAX) || j >= NB_2*pow(1 + (l>L_1DMAX)*REF_2, l - L_1DMAX) || z >= NB_3*pow(1 + REF_3, l)){
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
	#endif
	return n;
}

int calc_nlevels_tot(int l, int coord1, int coord2, int coord3) {
	int min_i, min_j, min_z;
	int max_i, max_j, max_z;
	int n_levels_tot, n_extra_cart; 

	//Determine if there is a nested-cartesian refinement level
	#if(GRID_CARTESIAN)
	if (l >= N_LEVELS_3D - 1) {
		//Calculate minimum index
		min_i = NB_1 * pow(1 + REF_1, l) / 2 - NB_1 * pow(1 + REF_1, N_LEVELS_3D - 2) / 2;
		min_j = NB_2 * pow(1 + REF_2, l) / 2 - NB_2 * pow(1 + REF_2, N_LEVELS_3D - 2) / 2;
		min_z = NB_3 * pow(1 + REF_3, l) / 2 - NB_3 * pow(1 + REF_3, N_LEVELS_3D - 2) / 2;

		//Calculate maximum index
		max_i = NB_1 * pow(1 + REF_1, l) / 2 + NB_1 * pow(1 + REF_1, N_LEVELS_3D - 2) / 2;
		max_j = NB_2 * pow(1 + REF_2, l) / 2 + NB_2 * pow(1 + REF_2, N_LEVELS_3D - 2) / 2;
		max_z = NB_3 * pow(1 + REF_3, l) / 2 + NB_3 * pow(1 + REF_3, N_LEVELS_3D - 2) / 2;
	}
	else {
		min_i = min_j = min_z = 0;
		max_i = NB_1 * pow(1 + REF_1, l);
		max_j = NB_2 * pow(1 + REF_2, l);
		max_z = NB_3 * pow(1 + REF_3, l);
	}

	//Calculate extra cartesian levels
	if (coord1 >= min_i && coord1 < max_i && coord2 >= min_j && coord2 < max_j && coord3 >= min_z && coord3 < max_z) {
		n_extra_cart = 1;
	}
	else {
		n_extra_cart = 0;
	}

	//Calculate total refinement levels (with a maximum cutoff)
	n_levels_tot = MY_MIN(MY_MAX((l + 1) + n_extra_cart, N_LEVELS_3D), N_LEVELS_3D + N_LEVELS_CART);
	#else
	n_levels_tot = N_LEVELS_3D;
	#endif

	return n_levels_tot;
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
	int n, n0, l, l_1D, l_3D, L_1DMAX, lc, i[N_LEVELS], j[N_LEVELS], z[N_LEVELS], keep_looping, index, offset;
	int max_level, i1, i2, i3, increment1, increment2, increment3, coord1, coord2, coord3, counter, n_levels_tot;
	#if(GRID_CARTESIAN)
	int n_extra_cart;
	int min_i, min_j, min_z;
	#endif

	//Initialize counters
	for (l = 0; l < N_LEVELS; l++) i[l] = j[l] = z[l] = 0;
	l_1D = l_3D = l = 0;
	n = n0  = 0;
	while (1){
		//Set level based on values from last iteration
		block[n][AMR_LEVEL1] = l_3D*REF_1;
		block[n][AMR_LEVEL2] = l_3D*REF_2;
		block[n][AMR_LEVEL3] = (l_1D + l_3D)*REF_3;
		l = l_1D + l_3D;
		block[n][AMR_LEVEL] = l;
	
		#if(Z_ORDER)
		if (l == 0){
			max_level = (int)(log((double)(MY_MAX(NB_1, MY_MAX(NB_2, NB_3)))) / log(2.)); //Gives the maximum 0-level of grid
			coord1 = 0;
			coord2 = 0;
			coord3 = 0;
			counter = n0;
			for (i1 = max_level; i1 >= 0; i1--){
				increment1 = MY_MIN(pow(2, i1), NB_1 - coord1 - 1);
				increment2 = MY_MIN(pow(2, i1), NB_2 - coord2 - 1);
				increment3 = MY_MIN(pow(2, i1), NB_3 - coord3 - 1);
				/*
				if (increment1 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= increment1*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3));
					coord1 += increment1;
				}
				if (increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= MY_MIN(pow(2, i1), (NB_1 - coord1))*increment2*MY_MIN(pow(2, i1), (NB_3 - coord3));
					if (increment1 == pow(2, i1)) coord1 -= increment1;
					coord2 += increment2;
				}
				if (increment1 == pow(2, i1) && increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= increment1*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3));
					coord1 += increment1;
				}
				if (increment3 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*increment3;
					if (increment1 == pow(2, i1)) coord1 -= increment1;
					if (increment2 == pow(2, i1)) coord2 -= increment2;
					coord3 += increment3;
				}
				if (increment3 == pow(2, i1) && increment1 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= increment1*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3));
					coord1 += increment1;
				}
				if (increment3 == pow(2, i1) && increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= MY_MIN(pow(2, i1), (NB_1 - coord1))*increment2*MY_MIN(pow(2, i1), (NB_3 - coord3));
					if (increment1 == pow(2, i1)) coord1 -= increment1;
					coord2 += increment2;
				}
				if (increment3 == pow(2, i1) && increment2 == pow(2, i1) && increment1 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_1 - coord1))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3))){
					counter -= increment1*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_3 - coord3));
					coord1 += increment1;
				}
				*/
				if (increment3 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= increment3*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1));
					coord3 += increment3;
				}
				if (increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= MY_MIN(pow(2, i1), (NB_3 - coord3))*increment2*MY_MIN(pow(2, i1), (NB_1 - coord1));
					if (increment3 == pow(2, i1)) coord3 -= increment3;
					coord2 += increment2;
				}
				if (increment3 == pow(2, i1) && increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= increment3*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1));
					coord3 += increment3;
				}
				if (increment1 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*increment1;
					if (increment3 == pow(2, i1)) coord3 -= increment3;
					if (increment2 == pow(2, i1)) coord2 -= increment2;
					coord1 += increment1;
				}
				if (increment1 == pow(2, i1) && increment3 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= increment3*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1));
					coord3 += increment3;
				}
				if (increment1 == pow(2, i1) && increment2 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= MY_MIN(pow(2, i1), (NB_3 - coord3))*increment2*MY_MIN(pow(2, i1), (NB_1 - coord1));
					if (increment3 == pow(2, i1)) coord3 -= increment3;
					coord2 += increment2;
				}
				if (increment1 == pow(2, i1) && increment2 == pow(2, i1) && increment3 == pow(2, i1) && counter >= MY_MIN(pow(2, i1), (NB_3 - coord3))*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1))) {
					counter -= increment3*MY_MIN(pow(2, i1), (NB_2 - coord2))*MY_MIN(pow(2, i1), (NB_1 - coord1));
					coord3 += increment3;
				}
				i[l] = coord1;
				j[l] = coord2;
				z[l] = coord3;
				if ((i[l] == NB_1) || (j[l] == NB_2) || (z[l] == NB_3)){
					fprintf(stderr, "Catastrophic error in grid mapping! \n");
					exit(0);
				}
			}
			n0++;
		}
		#endif

		if (!(DEREFINE_POLE == 0 || (NB_1 == 3 && N_LEVELS_1D == 0) || (NB_2 == 6  && N_LEVELS_1D == 1) || (NB_2 == 12 && N_LEVELS_1D == 2) || (NB_2 == 24 && N_LEVELS_1D == 3) || (NB_2 == 48 && N_LEVELS_1D == 4) || (NB_2 == 96  && N_LEVELS_1D == 5))){
			//if (rank == 0)fprintf(stderr, "For derefinement near the pole chose NB_2 3, 6, 12, 24, 48, 96 for 1, 2, 3, 4, 5 levels of derefinement near the pole! \n");
			//exit(0);
		}

		//Based on value of 0-th level block determine the number of 1D refinement levels
		#if(GRID_CARTESIAN)
		L_1DMAX = 0;
		#else
		if (j[0] < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)j[0] + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - j[0]) + 1.0) / log(2.)), N_LEVELS_1D);
		#endif

		//Set coordinates based on values from last iteration
		block[n][AMR_COORD1] = block[n][AMR_COORD2] = block[n][AMR_COORD3] = 0;
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++)	block[n][AMR_COORD1] += i[lc] * pow(1 + (block[n][AMR_LEVEL] >= L_1DMAX)*REF_1, block[n][AMR_LEVEL] - MY_MAX(lc, L_1DMAX));
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++)	block[n][AMR_COORD2] += j[lc] * pow(1 + (block[n][AMR_LEVEL] >= L_1DMAX)*REF_2, block[n][AMR_LEVEL] - MY_MAX(lc, L_1DMAX));
		for (lc = 0; lc <= block[n][AMR_LEVEL]; lc++){
			if (block[n][AMR_COORD2]/(int)pow(1+REF_2,block[n][AMR_LEVEL2]) == 0 && DEREFINE_POLE){
				block[n][AMR_LEVEL3] = ceil(log(block[n][AMR_COORD2] + 1) / log(2.))*REF_3;
				block[n][AMR_COORD3] += z[lc] * pow(1 + REF_3, MY_MIN(block[n][AMR_LEVEL] - lc, block[n][AMR_LEVEL3]));
			}
			else if (block[n][AMR_COORD2] / (int)pow(1 + REF_2, block[n][AMR_LEVEL2]) == (NB_2 - 1) && DEREFINE_POLE){
				block[n][AMR_LEVEL3] = ceil(log(NB_2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - block[n][AMR_COORD2]) / log(2.))*REF_3;
				block[n][AMR_COORD3] += z[lc] * pow(1 + REF_3, MY_MIN(block[n][AMR_LEVEL] - lc, block[n][AMR_LEVEL3]));
			}
			else{
				block[n][AMR_COORD3] += z[lc] * pow(1 + REF_3, block[n][AMR_LEVEL] - lc);
			}
		}

		//Store in array such that one can recover linear coordinate based on 4D coordinate
		offset = N_LEVELS_1D - L_1DMAX;
		#if(GRID_CARTESIAN)
		if (l >= N_LEVELS_3D) { 
			//Calculate minimum index
			min_i = NB_1 * pow(1 + REF_1, l) / 2 - NB_1 * pow(1 + REF_1, N_LEVELS_3D - 1) / 2;
			min_j = NB_2 * pow(1 + REF_2, l) / 2 - NB_2 * pow(1 + REF_2, N_LEVELS_3D - 1) / 2;
			min_z = NB_3 * pow(1 + REF_3, l) / 2 - NB_3 * pow(1 + REF_3, N_LEVELS_3D - 1) / 2;
		}
		else {
			min_i = min_j = min_z = 0;
		}
		if (block[n][AMR_COORD1] < min_i || block[n][AMR_COORD2] < min_j || block[n][AMR_COORD3] < min_z) {
			if (rank == 0) fprintf(stderr, "Error in gridding for Cartesian 1! \n");
			exit(0);
		}
		if (l >= N_LEVELS_3D) {
			index = (int)((block[n][AMR_COORD1] - min_i) * NB_3 * (int)pow(1 + REF_3, N_LEVELS_3D - 1) * NB_2 * pow(1 + REF_2, N_LEVELS_3D - 1) + (block[n][AMR_COORD2] - min_j) * NB_3 * pow(1 + REF_3, N_LEVELS_3D - 1) + (block[n][AMR_COORD3] - min_z));
		}
		else {
			index = (int)((block[n][AMR_COORD1] - min_i) * NB_3 * (int)pow(1 + REF_3, l) * NB_2 * pow(1 + REF_2, l) + (block[n][AMR_COORD2] - min_j) * NB_3 * pow(1 + REF_3, l) + (block[n][AMR_COORD3] - min_z));
		}
		#else
		index = (int)(block[n][AMR_COORD1] * NB_3*(int)pow(1 + REF_3, l + offset) * NB_2*pow(1 + REF_2*((l + offset) > N_LEVELS_1D), l - N_LEVELS_1D + offset) + block[n][AMR_COORD2] * NB_3*pow(1 + REF_3, l + offset) + block[n][AMR_COORD3]);
		#endif
		if (lin_coord[l + offset][index] != 0) fprintf(stderr, "Failure in setting grid! \n");
		lin_coord[l + offset][index] = n;
		lin_coord_RM[l + offset][index] = n;
		//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d L_1DMAX: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], L_1DMAX, block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);

		//Break out of loop if maximum block number reached and set n_max
		if (l == N_LEVELS_3D - 1 && block[n][AMR_COORD1] == NB_1*pow(1 + REF_1, l) - 1 && block[n][AMR_COORD2] == NB_2*pow(1 + REF_2, l) - 1 && block[n][AMR_COORD3] == NB_3*pow(1 + REF_3*(!DEREFINE_POLE), l) - 1){
			n_max = n;
			break;
		}

		//Calculate total number of levels in different block
		n_levels_tot= calc_nlevels_tot(l, block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);

		//Advance linear index by 1
		if (l < n_levels_tot + L_1DMAX - 1) {
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
					#if(!Z_ORDER)
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
					#endif
				}
				else{
					z[l]++;
					if (z[l] == 1 + REF_3*((block[n][AMR_COORD2] >= pow(1 + REF_2, block[n][AMR_LEVEL2] - l) && block[n][AMR_COORD2] < NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - pow(1 + REF_2, block[n][AMR_LEVEL2] - l)) || !DEREFINE_POLE)){
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

void check_cart_grid(void) {
	double delta_x1, delta_x2, delta_x3;
	double nr_cells1, nr_cells2, nr_cells3;
	double rinner;

	//Check if you have enough blocks for Cartesian grid
	if (NB_1 % 4 != 0 || NB_2 % 4 != 0 || NB_3 % 4 != 0) {
		if (rank == 0) fprintf(stderr, "Error 1 in setting up Cartesian grid: Check if you have enough blocks for Cartesian grid! \n");
		exit(0);
	}

	//Check if refinement happens in all 3 dimensions
	if (REF_1 + REF_2 + REF_3 != 3) {
		if (rank == 0) fprintf(stderr, "Error 2 in setting up Cartesian grid: Check if refinement happens in all 3 dimensions! \n");
		exit(0);
	}

	//You cannot use GRID_CARTESIAN together with DEREFINE_POLE
	if (DEREFINE_POLE) {
		if (rank == 0) fprintf(stderr, "Error 3 in setting up Cartesian grid: You cannot use GRID_CARTESIAN together with DEREFINE_POLE! \n");
		exit(0);
	}

	//You cannot have internal derefinement with Cartesian grid
	if (N_LEVELS_1D_INT != 0) {
		if (rank == 0) fprintf(stderr, "Error 4 in setting up Cartesian grid: You cannot have internal derefinement with Cartesian grid! \n");
		exit(0);
	}

	//You need to use asymmetric metric for Cartesian grid!
	if (NSY != 1) {
		if (rank == 0) fprintf(stderr, "Error 5 in setting up Cartesian grid: You need to use asymmetric metric for Cartesian grid! \n");
		exit(0);
	}

	//Calculate and print grid spacing near BH
	delta_x1 = 2 * ROUT / pow(1 + REF_1, N_LEVELS_CART) / (NB_1 * BS_1);
	delta_x2 = 2 * ROUT / pow(1 + REF_2, N_LEVELS_CART) / (NB_2 * BS_2);
	delta_x3 = 2 * ROUT / pow(1 + REF_3, N_LEVELS_CART) / (NB_3 * BS_3);
	if (rank == 0) fprintf(stderr, "Grid size (min) is: (%f x %f x %f) \n", delta_x1 * NB_1 * BS_1, delta_x2 * NB_2 * BS_2, delta_x3 * NB_3 * BS_3);
	if (rank == 0) fprintf(stderr, "Grid spacing (min) is: (%f x %f x %f) \n", delta_x1, delta_x2, delta_x3);
	
	//Do not allow refinement block to be within 1.1 times the event horizon radius
	rinner = 1.1*(1. + sqrt(1. - BH_SPIN * BH_SPIN));
	if (0.5 * delta_x1 * NB_1 * BS_1<rinner || 0.5 * delta_x2 * NB_2* BS_2 < rinner || 0.5 * delta_x3 * NB_3 * BS_3 < rinner) {
		if (rank == 0) fprintf(stderr, "Cartesian gridding error: Inner refinement block too small! n");
		exit(0);
	}

	//Calculate number of cells inside of event horizon
	nr_cells1 = 0.2 * (1. + sqrt(1. - BH_SPIN * BH_SPIN)) / (delta_x1);
	nr_cells2 = 0.2 * (1. + sqrt(1. - BH_SPIN * BH_SPIN)) / (delta_x2);
	nr_cells3 = 0.2 * (1. + sqrt(1. - BH_SPIN * BH_SPIN)) / (delta_x3);
	if (rank == 0) fprintf(stderr, "Nr of cells inside event horizon: (%f x %f x %f) \n", nr_cells1, nr_cells2, nr_cells3);

	//Calculate and print grid spacing far away from BH
	delta_x1 = 2 * ROUT / (NB_1 * BS_1);
	delta_x2 = 2 * ROUT / (NB_2 * BS_2);
	delta_x3 = 2 * ROUT / (NB_3 * BS_3);
	if (rank == 0) fprintf(stderr, "Grid size (max) is: (%f x %f x %f) \n", delta_x1 * NB_1 * BS_1, delta_x2 * NB_2 * BS_2, delta_x3 * NB_3 * BS_3);
	if (rank == 0) fprintf(stderr, "Grid spacing (max) is: (%f x %f x %f) \n", delta_x1, delta_x2, delta_x3);
}

//Sets the AMR hierarchy
void set_AMR(void){
	int n, n_parent, n_child[9], n_nbr[21], level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1,
		i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag;
	int y;

	#if(RAD_M1 || NEUTRINOS_M1)
	calc_ymax();
	#endif

	//Check if Cartesian grid is set up properly
	#if(GRID_CARTESIAN)
	check_cart_grid();
	#endif

	//Allocate arrays that are not block-specific and thus only need to be allocated at the start of a run and not between refinement steps
	block = (int(*)[NV])calloc(NB+1, sizeof(int[NV]));
	n_ord_node= (int(*)[NB_LOCAL])calloc(numtasks, sizeof(int[NB_LOCAL]));
	n_active_node= (int(*))calloc(numtasks, sizeof(int));

	#if(BOUND_TYPE2==TRANSMISSIVE)
	for (l = 0; l < N_LEVELS_3D; l++){
		E_avg1[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1+REF_1, l)*NB_3*pow(1+REF_3,l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg2[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg1_new[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));
		E_avg2_new[l] = (double(*)[BS_1 + 2 * N1G])calloc(NB_1*pow(1 + REF_1, l)*NB_3*pow(1 + REF_3, l), sizeof(double[BS_1 + 2 * N1G]));

	}
	#endif
	max_levels = 0;

	//Set memory flag to unallocated
	for (i = 0; i < NB_LOCAL; i++){
		mem_spot[i] = -1;
		mem_spot_gpu[i] = -1;
		mem_spot_gpu_bound[i] = -1;
	}

	//Set arrays for grid data output
	if (rank==(1 % numtasks)) {
		array_gdumpgrid = (int *)malloc((1 + NB*NV) * sizeof(int));
		array_rdumpgrid = (int *)malloc((1 + NB*NV) * sizeof(int));
	}


	for (l = 0; l < N_LEVELS; l++) {
		#if(GRID_CARTESIAN)
		lin_coord[l] = (int*)calloc(NB_1 * pow(1 + REF_1, MY_MIN(l, N_LEVELS_3D - 1)) * NB_2 * pow(1 + REF_2, MY_MIN(l, N_LEVELS_3D - 1)) * NB_3 * pow(1 + REF_3, MY_MIN(l, N_LEVELS_3D - 1)), sizeof(int));
		lin_coord_RM[l] = (int*)calloc(NB_1 * pow(1 + REF_1, MY_MIN(l, N_LEVELS_3D - 1)) * NB_2 * pow(1 +REF_2, MY_MIN(l, N_LEVELS_3D - 1)) * NB_3 * pow(1 + REF_3, MY_MIN(l, N_LEVELS_3D - 1)), sizeof(int));
		#else
		lin_coord[l] = (int *)calloc(NB_1*pow(1 + (l > N_LEVELS_1D)*REF_1, (l - N_LEVELS_1D))*NB_2*pow(1 + (l > N_LEVELS_1D)*REF_2, (l - N_LEVELS_1D))*NB_3*pow(1 + REF_3, l), sizeof(int));
		lin_coord_RM[l] = (int *)calloc(NB_1*pow(1 + (l > N_LEVELS_1D)*REF_1, (l - N_LEVELS_1D))*NB_2*pow(1 + (l > N_LEVELS_1D)*REF_2, (l - N_LEVELS_1D))*NB_3*pow(1 + REF_3, l), sizeof(int));
		#endif
	}

	//Set mapping from linear to 3D coordinates and vice-versa
	AMR_set_coord();

	//Indicates something is wrok with the grid mapping
	if (n_max != NB - 1){
		fprintf(stderr, "n_max: %d and NB: %d do not match! \n", n_max, NB);
		exit(0);
	}

	//Exit if incompatible grid for Cartesian runs
	if ((N_LEVELS_1D_INT > 0 || DEREFINE_POLE) && CARTESIAN) {
		fprintf(stderr, "Error: You cannot run with internal or external derefinement and cartesian grid! \n");
		exit(0);
	}

 	//Find parent for all blocks(refined and unrefined)
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		//Find parent of block
		if (block[n][AMR_LEVEL] == 0) block[n][AMR_PARENT] = -1; //-1 means no parent
		else{
			if (block[n][AMR_LEVEL1] > 0 || block[n][AMR_LEVEL2] > 0){
				new_level1 = block[n][AMR_LEVEL1] - REF_1;
				new_level2 = block[n][AMR_LEVEL2] - REF_2;
				new_level3 = block[n][AMR_LEVEL3] - REF_3 * ((block[n][AMR_COORD2] != 0 && block[n][AMR_COORD2] != NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) || !DEREFINE_POLE);
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
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);
		
		//Calculate the maximum number of refinement levels
		int n_levels_tot=calc_nlevels_tot(block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
		
		if (block[n][AMR_LEVEL] == n_levels_tot + L_1DMAX - 1){
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
			if (block[n][AMR_COORD2] == 0 && block[n][AMR_LEVEL] >= L_1DMAX && DEREFINE_POLE) {
				block[n][AMR_CHILD1] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3]);
				block[n][AMR_CHILD2] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3]);
				block[n][AMR_CHILD3] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD4] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD5] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3]);
				block[n][AMR_CHILD6] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3]);
				block[n][AMR_CHILD7] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD8] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			}
			else if (block[n][AMR_COORD2] == NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 && block[n][AMR_LEVEL] >= L_1DMAX && DEREFINE_POLE) {
				block[n][AMR_CHILD1] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD2] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD3] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3]);
				block[n][AMR_CHILD4] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3]);
				block[n][AMR_CHILD5] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD6] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD7] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3]);
				block[n][AMR_CHILD8] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3]);
			}
			else {
				block[n][AMR_CHILD1] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD2] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD3] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD4] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1), block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD5] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD6] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2), block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
				block[n][AMR_CHILD7] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3));
				block[n][AMR_CHILD8] = AMR_coord_linear2(block[n][AMR_LEVEL] + 1, j0, block[n][AMR_COORD1] * (1 + ref_1) + ref_1, block[n][AMR_COORD2] * (1 + ref_2) + ref_2, block[n][AMR_COORD3] * (1 + ref_3) + ref_3);
			}
		}
	}

	//Find neighbours of block
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
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
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		//First dimension is trivial
		if (i + 1 > i_max && BOUND_TYPE1==PERIODIC) i1 = 0;
		else if (i + 1 > i_max) i1 = -1;
		else i1 = i + 1;
		block[n][AMR_NBR2] = AMR_coord_linear2(level, j0, i1, j, z);

		if (i - 1 < 0 && BOUND_TYPE1 == PERIODIC) i1 = i_max;
		else i1 = i - 1;
		block[n][AMR_NBR4] = AMR_coord_linear2(level, j0, i1, j, z);

		//Third dimension is trivial
		if (z + 1 > z_max && BOUND_TYPE3 == PERIODIC) z1 = 0;
		else if (z + 1 > z_max) z1 = -1;
		else z1 = z + 1;
		block[n][AMR_NBR5] = AMR_coord_linear2(level, j0, i, j, z1);

		if (z - 1 < 0 && BOUND_TYPE3 == PERIODIC) z1 = z_max;
		else z1 = z - 1;
		block[n][AMR_NBR6] = AMR_coord_linear2(level, j0, i, j, z1);

		//Second dimension is not trivial
		if (j - 1 < 0 && BOUND_TYPE2 == PERIODIC) j1 = j_max;
		else j1 = j - 1;
		flag = 0;
		block[n][AMR_TAG1] = 0;
		if (DEREFINE_POLE){
			for (l = 0; l < MY_MIN(level, N_LEVELS_1D); l++){
				jbound = pow(1 + REF_2, l);
				if ((j == jbound * pow(1 + REF_2, block[n][AMR_LEVEL2])) || (j == (NB_2 - jbound) * pow(1 + REF_2, block[n][AMR_LEVEL2]))){
					block[n][AMR_NBR1] = NB;
					flag = 1;
					break;
				}

			}
			for (l = 1; l < N_LEVELS_3D; l++) {
				if (block[n][AMR_LEVEL2] >= l) {
					if ((j == pow(1 + REF_2, block[n][AMR_LEVEL2] - l)) || (j == (NB_2* pow(1 + REF_2, block[n][AMR_LEVEL2]) - pow(1 + REF_2, block[n][AMR_LEVEL2] - l)))) {
						block[n][AMR_NBR1] = NB;
						block[n][AMR_TAG1] = 1;
						flag = 1;
						break;
					}
				}
			}
		}
		if (flag == 0) block[n][AMR_NBR1] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z);

		if (j + 1 > j_max && BOUND_TYPE2 == PERIODIC) j1 = 0;
		else if (j + 1 > j_max) j1 = -1;
		else j1 = j + 1;
		flag = 0;
		block[n][AMR_TAG3] = 0;
		if (DEREFINE_POLE){
			for (l = 0; l < MY_MIN(level, N_LEVELS_1D); l++){
				jbound = pow(1 + REF_2, l);
				if (j == ((NB_2 - jbound)* pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) || (j == jbound* pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1)){
					block[n][AMR_NBR3] = NB;
					flag = 1;
					break;
				}
			}
			for (l = 1; l < N_LEVELS_3D; l++) {
				if (block[n][AMR_LEVEL2] >= l) {
					if ((j == pow(1 + REF_2, block[n][AMR_LEVEL2] - l) - 1) || (j == (NB_2* pow(1 + REF_2, block[n][AMR_LEVEL2]) - pow(1 + REF_2, block[n][AMR_LEVEL2] - l) - 1))) {
						block[n][AMR_NBR3] = NB;
						block[n][AMR_TAG3] = 3;
						flag = 1;
						break;
					}
				}
			}
		}
		if (flag == 0) block[n][AMR_NBR3] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z);

		//if (block[n][AMR_TAG1]==1)fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);


		//Set transmissive pole
		block[n][AMR_POLE] = 0; 
		#if (BOUND_TYPE2 == TRANSMISSIVE)
		//if (NB_3 % 2 != 0 && rank==0) fprintf(stderr, "Number of blocks in the third dimension is not an even number. This is incompatible with TRANS_BOUND");
		//Find the neighbours in the case we have transmissive boundary conditions at the pole
		if (j == 0){
			block[n][AMR_POLE] += 1;
			block[n][AMR_NBR1] = AMR_coord_linear2(level, j0, i, j, (z + NB_3*(int)pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2) % (z_max + 1));
		}
		if (j == j_max){
			block[n][AMR_POLE] += 2;
			block[n][AMR_NBR3] = AMR_coord_linear2(level, j0, i, j, (z + NB_3*(int)pow(1 + REF_3, block[n][AMR_LEVEL3]) / 2) % (z_max + 1));
		}
		#endif

	}

	//Find corners of block assuming only third dimension is periodic
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
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
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);

		//x-y plane
		//Corn 1
		if (i + 1 > i_max){
			if (BOUND_TYPE1 == PERIODIC) i1 = 0;
			else i1 = -1;
		}
		else i1 = i + 1;
		if (j - 1 < 0) {
			if (BOUND_TYPE2 == PERIODIC) j1 = j_max;
			else j1 = -1;
		}
		else  j1 = j - 1;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR2] >= 0 && DEREFINE_POLE) block[n][AMR_CORN1] = NB;
		else block[n][AMR_CORN1] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		//Corn 2
		if (i + 1 > i_max){
			if (BOUND_TYPE1 == PERIODIC) i1 = 0;
			else i1 = -1;
		}
		else i1 = i + 1;
		if (j + 1 > j_max) {
			if (BOUND_TYPE2 == PERIODIC) j1 = 0;
			else j1 = -1;
		}
		else j1 = j + 1;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR2] >= 0 && DEREFINE_POLE) block[n][AMR_CORN2] = NB;
		else block[n][AMR_CORN2] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		//Corn 3
		if (i - 1 < 0 ){
			if (BOUND_TYPE1 == PERIODIC) i1 = i_max;
			else i1 = -1;
		}
		else i1 = i - 1;
		if (j + 1 > j_max) {
			if (BOUND_TYPE2 == PERIODIC) j1 = 0;
			else j1 = -1;
		}
		else j1 = j + 1;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR4] >= 0 && DEREFINE_POLE) block[n][AMR_CORN3] = NB;
		else block[n][AMR_CORN3] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		//Corn 4
		if (i - 1 < 0){
			if (BOUND_TYPE1 == PERIODIC) i1 = i_max;
			else i1 = -1;
		}
		else i1 = i - 1;
		if (j - 1 < 0) {
			if (BOUND_TYPE2 == PERIODIC) j1 = j_max;
			else j1 = -1;
		}
		else j1 = j - 1;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR4] >= 0 && DEREFINE_POLE) block[n][AMR_CORN4] = NB;
		else block[n][AMR_CORN4] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i1, j1, z);

		//x-z plane
		//Corn 5
		if (i + 1 > i_max) {
			if (BOUND_TYPE1 == PERIODIC) i1 = 0;
			else i1 = -1;
		}
		else i1 = i + 1;
		if (z - 1 < 0) {
			if (BOUND_TYPE3 == PERIODIC) z1 = z_max;
			else z1 = -1;
		}
		else z1 = z - 1;
		block[n][AMR_CORN5] = AMR_coord_linear2(level, j0, i1, j, z1);

		//Corn 6
		if (i + 1 > i_max) {
			if (BOUND_TYPE1 == PERIODIC) i1 = 0;
			else i1 = -1;
		}
		else i1 = i + 1;
		if (z + 1 > z_max) {
			if (BOUND_TYPE3 == PERIODIC) z1 = 0;
			else z1 = -1;
		}
		else z1 = z + 1;
		block[n][AMR_CORN6] = AMR_coord_linear2(level, j0, i1, j, z1);

		//Corn 7
		if (i - 1 < 0) {
			if (BOUND_TYPE1 == PERIODIC) i1 = i_max;
			else i1 = -1;
		}
		else i1 = i - 1;
		if (z + 1 > z_max) {
			if (BOUND_TYPE3 == PERIODIC) z1 = 0;
			else z1 = -1;
		}
		else z1 = z + 1;
		block[n][AMR_CORN7] = AMR_coord_linear2(level, j0, i1, j, z1);

		//Corn 8
		if (i - 1 < 0) {
			if (BOUND_TYPE1 == PERIODIC) i1 = i_max;
			else i1 = -1;
		}
		else i1 = i - 1;
		if (z - 1 < 0) {
			if (BOUND_TYPE3 == PERIODIC) z1 = z_max;
			else z1 = -1;
		}
		else z1 = z - 1;
		block[n][AMR_CORN8] = AMR_coord_linear2(level, j0, i1, j, z1);

		//y-z plane
		//Corn 9
		if (j - 1 < 0) {
			if (BOUND_TYPE2 == PERIODIC) j1 = j_max;
			else j1 = -1;
		}
		else j1 = j - 1;
		if (z + 1 > z_max) {
			if (BOUND_TYPE3 == PERIODIC) z1 = 0;
			else z1 = -1;
		}
		else z1 = z + 1;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR5] >= 0 && DEREFINE_POLE) block[n][AMR_CORN9] = NB;
		else block[n][AMR_CORN9] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		//Corn 10
		if (j + 1 > j_max) {
			if (BOUND_TYPE2 == PERIODIC) j1 = 0;
			else j1 = -1;
		}
		else j1 = j + 1;
		if (z + 1 > z_max) {
			if (BOUND_TYPE3 == PERIODIC) z1 = 0;
			else z1 = -1;
		}
		else z1 = z + 1;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR5] >= 0 && DEREFINE_POLE) block[n][AMR_CORN10] = NB;
		else block[n][AMR_CORN10] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		//Corn 11
		if (j + 1 > j_max) {
			if (BOUND_TYPE2 == PERIODIC) j1 = 0;
			else j1 = -1;
		}
		else j1 = j + 1;
		if (z - 1 < 0) {
			if (BOUND_TYPE3 == PERIODIC) z1 = z_max;
			else z1 = -1;
		}
		else z1 = z - 1;
		if (block[n][AMR_NBR3] == NB && block[n][AMR_NBR6] >= 0 && DEREFINE_POLE) block[n][AMR_CORN11] = NB;
		else block[n][AMR_CORN11] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);

		//Corn 12
		if (j - 1 < 0) {
			if (BOUND_TYPE2 == PERIODIC) j1 = j_max;
			else j1 = -1;
		}
		else j1 = j - 1;
		if (z - 1 < 0) {
			if (BOUND_TYPE3 == PERIODIC) z1 = z_max;
			else z1 = -1;
		}
		else z1 = z - 1;
		if (block[n][AMR_NBR1] == NB && block[n][AMR_NBR6] >= 0 && DEREFINE_POLE) block[n][AMR_CORN12] = NB;
		else block[n][AMR_CORN12] = AMR_coord_linear2(level, (int)(j1 / pow(1 + REF_2, block[n][AMR_LEVEL2])), i, j1, z1);
	}
	
	//Set 2-way grid to negative
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		for (i = AMR_NBR1_3; i <= AMR_CORN12P; i++) block[n][i] = -1;
	}

	//Set 2-way grid hierarchy
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		if (j0 < NB_2 / 2) L_1DMAX = MY_MIN((ceil)(-0.001+log((double)j0 + 1.0) / log(2.)), N_LEVELS_1D);
		else L_1DMAX = MY_MIN((ceil)(-0.001+log((double)((NB_2 - 1) - j0) + 1.0) / log(2.)), N_LEVELS_1D);
		//Set NBR children
		if (block[n][AMR_NBR1] >= 0){
			if (block[n][AMR_NBR1] != NB){
				block[n][AMR_NBR1_3] = block[block[n][AMR_NBR1]][AMR_CHILD3];
				block[n][AMR_NBR1_4] = block[block[n][AMR_NBR1]][AMR_CHILD4];
				block[n][AMR_NBR1_7] = block[block[n][AMR_NBR1]][AMR_CHILD7];
				block[n][AMR_NBR1_8] = block[block[n][AMR_NBR1]][AMR_CHILD8];
			}
			else{
				#if(DEREFINE_POLE)
				//Difference in REF_1, REF_2
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					if (block[n][AMR_LEVEL1] < N_LEVELS_3D - 1){
						block[n][AMR_NBR1_3] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] * (1 + REF_1), block[n][AMR_COORD2] * (1 + REF_2) - 1, block[n][AMR_COORD3]);
						block[n][AMR_NBR1_4] = block[n][AMR_NBR1_3];
						block[n][AMR_NBR1_7] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] * (1 + REF_1) + REF_1, block[n][AMR_COORD2] * (1 + REF_2) - 1, block[n][AMR_COORD3]);
						block[n][AMR_NBR1_8] = block[n][AMR_NBR1_7];
					}
				}
				else{ //Difference in REF_3
					block[n][AMR_NBR1_3] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] * (1 + REF_3));
					block[n][AMR_NBR1_4] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] * (1 + REF_3) + 1);
					block[n][AMR_NBR1_7] = block[n][AMR_NBR1_3];
					block[n][AMR_NBR1_8] = block[n][AMR_NBR1_4];
				}
				#elif(GRID_CARTESIAN)
				block[n][AMR_NBR1_3] = -1;
				block[n][AMR_NBR1_4] = -1;
				block[n][AMR_NBR1_7] = -1;
				block[n][AMR_NBR1_8] = -1;
				#endif
			}
		}
		if (block[n][AMR_NBR2] >= 0){
			if (block[n][AMR_NBR2] != NB) {
				block[n][AMR_NBR2_1] = block[block[n][AMR_NBR2]][AMR_CHILD1];
				block[n][AMR_NBR2_2] = block[block[n][AMR_NBR2]][AMR_CHILD2];
				block[n][AMR_NBR2_3] = block[block[n][AMR_NBR2]][AMR_CHILD3];
				block[n][AMR_NBR2_4] = block[block[n][AMR_NBR2]][AMR_CHILD4];
			}
			else {
				block[n][AMR_NBR2_1] = -1;
				block[n][AMR_NBR2_2] = -1;
				block[n][AMR_NBR2_3] = -1;
				block[n][AMR_NBR2_4] = -1;
			}
		}
		if (block[n][AMR_NBR3] >= 0){
			if (block[n][AMR_NBR3] != NB){
				block[n][AMR_NBR3_1] = block[block[n][AMR_NBR3]][AMR_CHILD1];
				block[n][AMR_NBR3_2] = block[block[n][AMR_NBR3]][AMR_CHILD2];
				block[n][AMR_NBR3_5] = block[block[n][AMR_NBR3]][AMR_CHILD5];
				block[n][AMR_NBR3_6] = block[block[n][AMR_NBR3]][AMR_CHILD6];
			}
			else{
				#if(DEREFINE_POLE)
				//Difference in REF_3
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Orig: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
					block[n][AMR_NBR3_1] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] * (1 + REF_3));
					block[n][AMR_NBR3_2] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] * (1 + REF_3) + 1);
					block[n][AMR_NBR3_5] = block[n][AMR_NBR3_1];
					block[n][AMR_NBR3_6] = block[n][AMR_NBR3_2];
					//int test = block[n][AMR_NBR3_1];
					//if (block[n][AMR_COORD1] == 0 && block[n][AMR_COORD3] == 0)fprintf(stderr, "Child: n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d \n", test, block[test][AMR_LEVEL], block[test][AMR_LEVEL1], block[test][AMR_LEVEL2], block[test][AMR_LEVEL3], block[test][AMR_COORD1], block[test][AMR_COORD2], block[test][AMR_COORD3]);
				}
				else{ //Difference in REF_1, REF_2
					if (block[n][AMR_LEVEL1] < N_LEVELS_3D - 1){
						block[n][AMR_NBR3_1] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] * (1 + REF_1), (block[n][AMR_COORD2] + 1) * (1 + REF_2), block[n][AMR_COORD3]);
						block[n][AMR_NBR3_2] = block[n][AMR_NBR3_1];
						block[n][AMR_NBR3_5] = AMR_coord_linear2(block[n][AMR_LEVEL] + (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] * (1 + REF_1) + REF_1, (block[n][AMR_COORD2] + 1) * (1 + REF_2), block[n][AMR_COORD3]);
						block[n][AMR_NBR3_6] = block[n][AMR_NBR3_5];
					}
				}
				#elif(GRID_CARTESIAN)
				block[n][AMR_NBR3_1] = -1;
				block[n][AMR_NBR3_2] = -1;
				block[n][AMR_NBR3_5] = -1;
				block[n][AMR_NBR3_6] = -1;
				#endif
			}
		}
		if (block[n][AMR_NBR4] >= 0){
			if (block[n][AMR_NBR4] != NB) {
				block[n][AMR_NBR4_5] = block[block[n][AMR_NBR4]][AMR_CHILD5];
				block[n][AMR_NBR4_6] = block[block[n][AMR_NBR4]][AMR_CHILD6];
				block[n][AMR_NBR4_7] = block[block[n][AMR_NBR4]][AMR_CHILD7];
				block[n][AMR_NBR4_8] = block[block[n][AMR_NBR4]][AMR_CHILD8];
			}
			else {
				block[n][AMR_NBR4_5] = -1;
				block[n][AMR_NBR4_6] = -1;
				block[n][AMR_NBR4_7] = -1;
				block[n][AMR_NBR4_8] = -1;
			}
		}
		if (block[n][AMR_NBR5] >= 0){
			if (block[n][AMR_NBR5] != NB) {
				block[n][AMR_NBR5_1] = block[block[n][AMR_NBR5]][AMR_CHILD1];
				block[n][AMR_NBR5_3] = block[block[n][AMR_NBR5]][AMR_CHILD3];
				block[n][AMR_NBR5_5] = block[block[n][AMR_NBR5]][AMR_CHILD5];
				block[n][AMR_NBR5_7] = block[block[n][AMR_NBR5]][AMR_CHILD7];
			}
			else {
				block[n][AMR_NBR5_1] = -1;
				block[n][AMR_NBR5_3] = -1;
				block[n][AMR_NBR5_5] = -1;
				block[n][AMR_NBR5_7] = -1;
			}
		}
		if (block[n][AMR_NBR6] >= 0){
			if (block[n][AMR_NBR6] != NB) {
				block[n][AMR_NBR6_2] = block[block[n][AMR_NBR6]][AMR_CHILD2];
				block[n][AMR_NBR6_4] = block[block[n][AMR_NBR6]][AMR_CHILD4];
				block[n][AMR_NBR6_6] = block[block[n][AMR_NBR6]][AMR_CHILD6];
				block[n][AMR_NBR6_8] = block[block[n][AMR_NBR6]][AMR_CHILD8];
			}
			else {
				block[n][AMR_NBR6_2] = -1;
				block[n][AMR_NBR6_4] = -1;
				block[n][AMR_NBR6_6] = -1;
				block[n][AMR_NBR6_8] = -1;
			}
		}
	}
	
	//Set corn children
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		if (block[n][AMR_CORN1] >= 0){
			if (block[n][AMR_CORN1] != NB){
				block[n][AMR_CORN1_1] = block[block[n][AMR_CORN1]][AMR_CHILD3];
				block[n][AMR_CORN1_2] = block[block[n][AMR_CORN1]][AMR_CHILD4];
			}
			else if(DEREFINE_POLE){
				if (block[n][AMR_NBR1_7] >= 0) block[n][AMR_CORN1_1] = block[block[n][AMR_NBR1_7]][AMR_NBR2];
				if (block[n][AMR_NBR1_8] >= 0) block[n][AMR_CORN1_2] = block[block[n][AMR_NBR1_8]][AMR_NBR2];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN1_1] = -1;
				block[n][AMR_CORN1_2] = -1;
			}
		}
		if (block[n][AMR_CORN2] >= 0){
			if (block[n][AMR_CORN2] != NB){
				block[n][AMR_CORN2_1] = block[block[n][AMR_CORN2]][AMR_CHILD1];
				block[n][AMR_CORN2_2] = block[block[n][AMR_CORN2]][AMR_CHILD2];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR3_5] >= 0) block[n][AMR_CORN2_1] = block[block[n][AMR_NBR3_5]][AMR_NBR2];
				if (block[n][AMR_NBR3_6] >= 0) block[n][AMR_CORN2_2] = block[block[n][AMR_NBR3_6]][AMR_NBR2];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN2_1] = -1;
				block[n][AMR_CORN2_2] = -1;
			}
		}
		if (block[n][AMR_CORN3] >= 0){
			if (block[n][AMR_CORN3] != NB){
				block[n][AMR_CORN3_1] = block[block[n][AMR_CORN3]][AMR_CHILD5];
				block[n][AMR_CORN3_2] = block[block[n][AMR_CORN3]][AMR_CHILD6];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR3_1] >= 0) block[n][AMR_CORN3_1] = block[block[n][AMR_NBR3_1]][AMR_NBR4];
				if (block[n][AMR_NBR3_2] >= 0) block[n][AMR_CORN3_2] = block[block[n][AMR_NBR3_2]][AMR_NBR4];
			}
			else if (GRID_CARTESIAN){
				block[n][AMR_CORN3_1] = -1;
				block[n][AMR_CORN3_2] = -1;
			}
		}
		if (block[n][AMR_CORN4] >= 0){
			if (block[n][AMR_CORN4] != NB){
				block[n][AMR_CORN4_1] = block[block[n][AMR_CORN4]][AMR_CHILD7];
				block[n][AMR_CORN4_2] = block[block[n][AMR_CORN4]][AMR_CHILD8];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR1_3] >= 0) block[n][AMR_CORN4_1] = block[block[n][AMR_NBR1_3]][AMR_NBR4];
				if (block[n][AMR_NBR1_4] >= 0) block[n][AMR_CORN4_2] = block[block[n][AMR_NBR1_4]][AMR_NBR4];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN4_1] = -1;
				block[n][AMR_CORN4_2] = -1;
			}
		}

		if (block[n][AMR_CORN5] >= 0){
			if (block[n][AMR_CORN5] != NB) {
				block[n][AMR_CORN5_1] = block[block[n][AMR_CORN5]][AMR_CHILD2];
				block[n][AMR_CORN5_2] = block[block[n][AMR_CORN5]][AMR_CHILD4];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN5_1] = -1;
				block[n][AMR_CORN5_2] = -1;
			}
		}
		if (block[n][AMR_CORN6] >= 0){
			if (block[n][AMR_CORN6] != NB) {
				block[n][AMR_CORN6_1] = block[block[n][AMR_CORN6]][AMR_CHILD1];
				block[n][AMR_CORN6_2] = block[block[n][AMR_CORN6]][AMR_CHILD3];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN6_1] = -1;
				block[n][AMR_CORN6_2] = -1;
			}
		}
		if (block[n][AMR_CORN7] >= 0){
			if (block[n][AMR_CORN7] != NB) {
				block[n][AMR_CORN7_1] = block[block[n][AMR_CORN7]][AMR_CHILD5];
				block[n][AMR_CORN7_2] = block[block[n][AMR_CORN7]][AMR_CHILD7];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN7_1] = -1;
				block[n][AMR_CORN7_2] = -1;
			}
		}
		if (block[n][AMR_CORN8] >= 0){
			if (block[n][AMR_CORN8] != NB) {
				block[n][AMR_CORN8_1] = block[block[n][AMR_CORN8]][AMR_CHILD6];
				block[n][AMR_CORN8_2] = block[block[n][AMR_CORN8]][AMR_CHILD8];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN8_1] = -1;
				block[n][AMR_CORN8_2] = -1;
			}
		}

		if (block[n][AMR_CORN9] >= 0){
			if (block[n][AMR_CORN9] != NB){
				block[n][AMR_CORN9_1] = block[block[n][AMR_CORN9]][AMR_CHILD3];
				block[n][AMR_CORN9_2] = block[block[n][AMR_CORN9]][AMR_CHILD7];
			}
			else if(DEREFINE_POLE){
				if (block[n][AMR_NBR1_4] >= 0) block[n][AMR_CORN9_1] = block[block[n][AMR_NBR1_4]][AMR_NBR5];
				if (block[n][AMR_NBR1_8] >= 0) block[n][AMR_CORN9_2] = block[block[n][AMR_NBR1_8]][AMR_NBR5];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN9_1] = -1;
				block[n][AMR_CORN9_2] = -1;
			}
		}
		if (block[n][AMR_CORN10] >= 0){
			if (block[n][AMR_CORN10] != NB){
				block[n][AMR_CORN10_1] = block[block[n][AMR_CORN10]][AMR_CHILD1];
				block[n][AMR_CORN10_2] = block[block[n][AMR_CORN10]][AMR_CHILD5];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR3_2] >= 0) block[n][AMR_CORN10_1] = block[block[n][AMR_NBR3_2]][AMR_NBR5];
				if (block[n][AMR_NBR3_6] >= 0) block[n][AMR_CORN10_2] = block[block[n][AMR_NBR3_6]][AMR_NBR5];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN10_1] = -1;
				block[n][AMR_CORN10_2] = -1;
			}
		}
		if (block[n][AMR_CORN11] >= 0){
			if (block[n][AMR_CORN11] != NB){
				block[n][AMR_CORN11_1] = block[block[n][AMR_CORN11]][AMR_CHILD2];
				block[n][AMR_CORN11_2] = block[block[n][AMR_CORN11]][AMR_CHILD6];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR3_1] >= 0) block[n][AMR_CORN11_1] = block[block[n][AMR_NBR3_1]][AMR_NBR6];
				if (block[n][AMR_NBR3_5] >= 0) block[n][AMR_CORN11_2] = block[block[n][AMR_NBR3_5]][AMR_NBR6];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN11_1] = -1;
				block[n][AMR_CORN11_2] = -1;
			}
		}
		if (block[n][AMR_CORN12] >= 0){
			if (block[n][AMR_CORN12] != NB){
				block[n][AMR_CORN12_1] = block[block[n][AMR_CORN12]][AMR_CHILD4];
				block[n][AMR_CORN12_2] = block[block[n][AMR_CORN12]][AMR_CHILD8];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_NBR1_3] >= 0) block[n][AMR_CORN12_1] = block[block[n][AMR_NBR1_3]][AMR_NBR6];
				if (block[n][AMR_NBR1_7] >= 0) block[n][AMR_CORN12_2] = block[block[n][AMR_NBR1_7]][AMR_NBR6];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN12_1] = -1;
				block[n][AMR_CORN12_2] = -1;
			}
		}
	}

	//Set NBR parent
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));
		//fprintf(stderr, "n1: %d level: %d level1: %d level2: %d level3: %d i: %d j: %d z: %d tag: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_LEVEL1], block[n][AMR_LEVEL2], block[n][AMR_LEVEL3], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3], block[n][AMR_TAG1]);

		if (block[n][AMR_NBR1] >= 0){
			if (block[n][AMR_NBR1] != NB) {
				block[n][AMR_NBR1P] = block[block[n][AMR_NBR1]][AMR_PARENT];
			}
			else if(DEREFINE_POLE){
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_NBR1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					//block[n][AMR_NBR1P] = block[block[n][AMR_NBR1_3]][AMR_PARENT];
					block[n][AMR_NBR1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_NBR2] >= 0){
			if (block[n][AMR_NBR2] != NB) {
				block[n][AMR_NBR2P] = block[block[n][AMR_NBR2]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR2P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] + 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_NBR3] >= 0){
			if (block[n][AMR_NBR3] != NB){
				block[n][AMR_NBR3P] = block[block[n][AMR_NBR3]][AMR_PARENT];
			}
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){
					//block[n][AMR_NBR3P] = block[block[n][AMR_NBR3_1]][AMR_PARENT];
					block[n][AMR_NBR3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
				else{ //Difference in REF_3
					block[n][AMR_NBR3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_NBR4] >= 0){
			if (block[n][AMR_NBR4] != NB) {
				block[n][AMR_NBR4P] = block[block[n][AMR_NBR4]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR4P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] - 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_NBR5] >= 0){
			if (block[n][AMR_NBR5] != NB) {
				block[n][AMR_NBR5P] = block[block[n][AMR_NBR5]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR5P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] + 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_NBR6] >= 0){
			if (block[n][AMR_NBR6] != NB) {
				block[n][AMR_NBR6P] = block[block[n][AMR_NBR6]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_NBR6P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] - 1) / (1 + REF_3));
			}
		}
	}

	//Set corn parent
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		//Calculate the maximum level of the block at the given location
		j0 = (int)(block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]));

		if (block[n][AMR_CORN1] >= 0){
			if (block[n][AMR_CORN1] != NB)block[n][AMR_CORN1P] = block[block[n][AMR_CORN1]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] + 1, block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					//if (block[n][AMR_CORN1_1] >= 0)block[n][AMR_CORN1P] = block[block[n][AMR_CORN1_1]][AMR_PARENT];
					block[n][AMR_CORN1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), (block[n][AMR_COORD1] + 1) / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN1P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] + 1) / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN2] >= 0){
			if (block[n][AMR_CORN2] != NB)block[n][AMR_CORN2P] = block[block[n][AMR_CORN2]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					//if (block[n][AMR_CORN2_1] >= 0)block[n][AMR_CORN2P] = block[block[n][AMR_CORN2_1]][AMR_PARENT];
					block[n][AMR_CORN2P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), (block[n][AMR_COORD1] + 1) / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
				else{
					block[n][AMR_CORN2P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] + 1, block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN2P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] + 1) / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN3] >= 0){
			if (block[n][AMR_CORN3] != NB)block[n][AMR_CORN3P] = block[block[n][AMR_CORN3]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					//if (block[n][AMR_CORN3_1] >= 0)block[n][AMR_CORN3P] = block[block[n][AMR_CORN3_1]][AMR_PARENT];
					block[n][AMR_CORN3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), (block[n][AMR_COORD1] - 1) / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
				else{
					block[n][AMR_CORN3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] - 1, block[n][AMR_COORD2] + 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN3P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] - 1) / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN4] >= 0){
			if (block[n][AMR_CORN4] != NB)block[n][AMR_CORN4P] = block[block[n][AMR_CORN4]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					block[n][AMR_CORN4P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] - 1, block[n][AMR_COORD2] - 1, block[n][AMR_COORD3] / (1 + REF_3));
				}
				else{
					//if (block[n][AMR_CORN4_1] >= 0)block[n][AMR_CORN4P] = block[block[n][AMR_CORN4_1]][AMR_PARENT];
					block[n][AMR_CORN4P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), (block[n][AMR_COORD1] - 1) / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3]);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN4P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] - 1) / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), block[n][AMR_COORD3] / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN5] >= 0) {
			if (block[n][AMR_CORN5] != NB){
				block[n][AMR_CORN5P] = block[block[n][AMR_CORN5]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN5P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] + 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] - 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN6] >= 0){
			if (block[n][AMR_CORN6] != NB) {
				block[n][AMR_CORN6P] = block[block[n][AMR_CORN6]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN6P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] + 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] + 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN7] >= 0){
			if (block[n][AMR_CORN7] != NB) {
				block[n][AMR_CORN7P] = block[block[n][AMR_CORN7]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN7P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] - 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] + 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN8] >= 0){
			if (block[n][AMR_CORN8] != NB) {
				block[n][AMR_CORN8P] = block[block[n][AMR_CORN8]][AMR_PARENT];
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN8P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, (block[n][AMR_COORD1] - 1) / (1 + REF_1), block[n][AMR_COORD2] / (1 + REF_2), (block[n][AMR_COORD3] - 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN9] >= 0){
			if (block[n][AMR_CORN9] != NB)block[n][AMR_CORN9P] = block[block[n][AMR_CORN9]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					z = (block[n][AMR_COORD3] + REF_3) / (1 + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3] - 1) - 1;
					if (z > z_max && BOUND_TYPE3 == PERIODIC) z = 0;
					block[n][AMR_CORN9P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, z);
				}
				else{
					//if (block[n][AMR_CORN9_1] >= 0)block[n][AMR_CORN9P] = block[block[n][AMR_CORN9_1]][AMR_PARENT];
					z = (block[n][AMR_COORD3] + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;
					if (z > z_max && BOUND_TYPE3 == PERIODIC) z = 0;
					block[n][AMR_CORN9P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), z);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN9P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), (block[n][AMR_COORD3] + 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN10] >= 0){
			if (block[n][AMR_CORN10] != NB)block[n][AMR_CORN10P] = block[block[n][AMR_CORN10]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3	
					//if (block[n][AMR_CORN10_1] >= 0)block[n][AMR_CORN10P] = block[block[n][AMR_CORN10_1]][AMR_PARENT];
					z = (block[n][AMR_COORD3] + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;
					if (z > z_max && BOUND_TYPE3 == PERIODIC) z = 0;
					block[n][AMR_CORN10P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), z);
				}
				else{
					z = (block[n][AMR_COORD3] + REF_3) / (1 + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3] - 1) - 1;
					if (z > z_max && BOUND_TYPE3 == PERIODIC) z = 0;
					block[n][AMR_CORN10P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, z);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN10P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), (block[n][AMR_COORD3] + 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN11] >= 0){
			if (block[n][AMR_CORN11] != NB)block[n][AMR_CORN11P] = block[block[n][AMR_CORN11]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					//if (block[n][AMR_CORN11_1] >= 0)block[n][AMR_CORN11P] = block[block[n][AMR_CORN11_1]][AMR_PARENT];
					z = (block[n][AMR_COORD3] - REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;
					if (block[n][AMR_COORD3] - REF_3 < 0 && BOUND_TYPE3 == PERIODIC) z = z_max;
					block[n][AMR_CORN11P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] == 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), z);
				}
				else{
					z = (block[n][AMR_COORD3] - REF_3) / (1 + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3] - 1) - 1;
					if (block[n][AMR_COORD3] - REF_3 < 0 && BOUND_TYPE3 == PERIODIC) z = z_max;
					block[n][AMR_CORN11P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG3] != 3), j0 + (block[n][AMR_TAG3] != 3), block[n][AMR_COORD1], block[n][AMR_COORD2] + 1, z);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN11P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] + 1) / (1 + REF_2), (block[n][AMR_COORD3] - 1) / (1 + REF_3));
			}
		}
		if (block[n][AMR_CORN12] >= 0){
			if (block[n][AMR_CORN12] != NB)block[n][AMR_CORN12P] = block[block[n][AMR_CORN12]][AMR_PARENT];
			else if (DEREFINE_POLE) {
				if (block[n][AMR_COORD2] < NB_2 / 2 * pow(1 + REF_2, block[n][AMR_LEVEL2])){ //Difference in REF_3
					z = (block[n][AMR_COORD3] - REF_3) / (1 + REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3] - 1) - 1;
					if (block[n][AMR_COORD3] - REF_3 < 0 && BOUND_TYPE3 == PERIODIC) z = z_max;
					block[n][AMR_CORN12P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] != 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1], block[n][AMR_COORD2] - 1, z);
				}
				else{
					//if (block[n][AMR_CORN12_1] >= 0)block[n][AMR_CORN12P] = block[block[n][AMR_CORN12_1]][AMR_PARENT];
					z = (block[n][AMR_COORD3] - REF_3);
					z_max = NB_3*pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1;
					if (block[n][AMR_COORD3] - REF_3 < 0 && BOUND_TYPE3 == PERIODIC) z = z_max;
					block[n][AMR_CORN12P] = AMR_coord_linear2(block[n][AMR_LEVEL] - (block[n][AMR_TAG1] == 1), j0 - (block[n][AMR_TAG1] != 1), block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), z);
				}
			}
			else if (GRID_CARTESIAN) {
				block[n][AMR_CORN12P] = AMR_coord_linear2(block[n][AMR_LEVEL] - 1, j0, block[n][AMR_COORD1] / (1 + REF_1), (block[n][AMR_COORD2] - 1) / (1 + REF_2), (block[n][AMR_COORD3] - 1) / (1 + REF_3));
			}
		}
	}

	//Reset fake parent corners
	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		if (block[n][AMR_CORN1P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN1P]][AMR_LEVEL1];
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN1P]][AMR_LEVEL2];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == ref_1 && block[n][AMR_COORD2] % (1 + ref_2) == 0))block[n][AMR_CORN1P] = -1;
		}
		if (block[n][AMR_CORN2P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN2P]][AMR_LEVEL1];
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN2P]][AMR_LEVEL2];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == ref_1 && block[n][AMR_COORD2] % (1 + ref_2) == ref_2))block[n][AMR_CORN2P] = -1;
		}
		if (block[n][AMR_CORN3P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN3P]][AMR_LEVEL1];
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN3P]][AMR_LEVEL2];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == 0 && block[n][AMR_COORD2] % (1 + ref_2) == ref_2))block[n][AMR_CORN3P] = -1;
		}
		if (block[n][AMR_CORN4P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN4P]][AMR_LEVEL1];
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN4P]][AMR_LEVEL2];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == 0 && block[n][AMR_COORD2] % (1 + ref_2) == 0))block[n][AMR_CORN4P] = -1;
		}
		if (block[n][AMR_CORN5P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN5P]][AMR_LEVEL1];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN5P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == ref_1 && block[n][AMR_COORD3] % (1 + ref_3) == 0))block[n][AMR_CORN5P] = -1;
		}
		if (block[n][AMR_CORN6P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN6P]][AMR_LEVEL1];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN6P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == ref_1 && block[n][AMR_COORD3] % (1 + ref_3) == ref_3))block[n][AMR_CORN6P] = -1;
		}
		if (block[n][AMR_CORN7P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN7P]][AMR_LEVEL1];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN7P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == 0 && block[n][AMR_COORD3] % (1 + ref_3) == ref_3))block[n][AMR_CORN7P] = -1;
		}
		if (block[n][AMR_CORN8P] >= 0){
			ref_1 = block[n][AMR_LEVEL1] - block[block[n][AMR_CORN8P]][AMR_LEVEL1];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN8P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD1] % (1 + ref_1) == 0 && block[n][AMR_COORD3] % (1 + ref_3) == 0))block[n][AMR_CORN8P] = -1;
		}
		if (block[n][AMR_CORN9P] >= 0){
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN9P]][AMR_LEVEL2];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN9P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD2] % (1 + ref_2) == 0 && block[n][AMR_COORD3] % (1 + ref_3) == ref_3))block[n][AMR_CORN9P] = -1;
		}
		if (block[n][AMR_CORN10P] >= 0){
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN10P]][AMR_LEVEL2];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN10P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD2] % (1 + ref_2) == ref_2 && block[n][AMR_COORD3] % (1 + ref_3) == ref_3))block[n][AMR_CORN10P] = -1;
		}
		if (block[n][AMR_CORN11P] >= 0){
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN11P]][AMR_LEVEL2];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN11P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD2] % (1 + ref_2) == ref_2 && block[n][AMR_COORD3] % (1 + ref_3) == 0))block[n][AMR_CORN11P] = -1;
		}
		if (block[n][AMR_CORN12P] >= 0){
			ref_2 = block[n][AMR_LEVEL2] - block[block[n][AMR_CORN12P]][AMR_LEVEL2];
			ref_3 = block[n][AMR_LEVEL3] - block[block[n][AMR_CORN12P]][AMR_LEVEL3];
			if (!(block[n][AMR_COORD2] % (1 + ref_2) == 0 && block[n][AMR_COORD3] % (1 + ref_3) == 0))block[n][AMR_CORN12P] = -1;
		}
	}

	//Set offsets and size of blocks#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		N1_GPU_offset[n] = block[n][AMR_COORD1] * BS_1;
		N2_GPU_offset[n] = block[n][AMR_COORD2] * BS_2;
		N3_GPU_offset[n] = block[n][AMR_COORD3] * BS_3;	
	}

	#pragma omp parallel for schedule(static,  NB/nthreads) private(n, n_parent, n_child, n_nbr, level, level1, level2, level3, new_level1, new_level2, new_level3, i, j,j0, z, l, i1, j1, z1, i_max, j_max, z_max, i_parent, j_parent, z_parent, ind, ref_1, ref_2, ref_3, L_1DMAX, jbound, flag, y)
	for (n = 0; n <= n_max; n++){
		set_points(n);

		//For the moment don't refine any block
		block[n][AMR_REFINED] = 0;

		block[n][GDUMP_WRITTEN] = 0;
		block[n][GDUMP_WRITTEN_REDUCED] = 0;

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

		//Give blocks near pole extra weight for load balancing
		if (block[n][AMR_POLE] == 0)block[n][AMR_WEIGHT] = 1;
		else block[n][AMR_WEIGHT] = MAX_WEIGHT;
	}

	if (BS_3 / (int)pow(2, N_LEVELS_1D_INT)<4 && N_LEVELS_1D_INT > 0){
		if (rank == 0) fprintf(stderr, "Grid too small for number of internal derefinement levels! \n");
		//exit(0);
	}

	if (BS_2 % (int)pow(2, N_LEVELS_1D_INT) != 0 || BS_3 % (int)pow(2, N_LEVELS_1D_INT) != 0){
		if (rank == 0) fprintf(stderr, "Grid not power of 2 of internal derefinment levels! \n");
		//exit(0);
	}

	#if(DUMP_SMALL)
	if ((BS_1%REDUCE_FACTOR1 != 0 || BS_2%REDUCE_FACTOR2 != 0 || BS_3%REDUCE_FACTOR3 != 0) && DUMP_SMALL) {
		if (rank == 0) fprintf(stderr, "Grid reduction incompatible with grid size! \n");
		exit(0);
	}
	#endif

	//Grid parameters
	set_gridparam();

	//Check if there is a restart file with the preset grid hierarchy
	restart_read_param();

	//Activate all blocks in arrays n_ord and n_ord_total
	activate_blocks();

	#if(GPU_ENABLED)
	//Calculate memory consumption on each GPU
	double total_mem = 0.;

	#if(CUDA_MEMCALC)
	for (n = 0; n < n_active_total; n++) {
		total_mem += calc_mem_gpu(n_ord_total[n]) / pow(10., 9.);
	}
	double mem_per_block = total_mem / n_active_total;
	max_blocks = (int)(n_active_total + (numtasks * gpu_mem - total_mem) / mem_per_block);
	#else
	for (n = 0; n < n_active_total; n++) {
		total_mem += calc_mem_gpu(n_ord_total[n]) / pow(10., 9.);
	}
	double mem_per_block = total_mem / n_active_total;
	max_blocks = (int)(n_active_total + (numtasks * GPU_MEM - total_mem) / mem_per_block);
	#endif

	if (max_blocks < n_active_total) {
		if (rank == 0 ) fprintf(stderr, "Too little GPU memory. Max_blocks: %d %d %f Quiting! \n", max_blocks, n_active, total_mem);
		//exit(0);
	}
	#else
	double total_mem = 0.;
	for (n = 0; n < n_active_total; n++) {
		total_mem += calc_mem_cpu(n_ord_total[n]) / pow(10., 9.);
	}
	double mem_per_block = total_mem / n_active_total;
	max_blocks = (int)(n_active_total + (numtasks * CPU_MEM - total_mem) / mem_per_block);

	if (max_blocks< n_active_total) {
		if (rank == 0) fprintf(stderr, "Too little CPU memory. Max_blocks: %d Quiting! \n", max_blocks);
	}
	#endif

	balance_load();
	for (n = 0; n < n_active; n++) {
		alloc_bounds_GPU(n_ord[n]);
	}

	set_corners(0);
}

void balance_load(void){

	int i,j,g, z, node, tt, fp, ip, y, rem, nr_timesteps, n_active_localsteps[NB];
	int i1, j1, z1, k, n, u, b, stride, count=0;
	int n_active_total_steps = 0, n_active_total_steps_t[10], steps_total_RM[NB];
	int NODE[NB], GPU[NB], temp;
	int n_active_total_t[10], (*n_ord_total_RM_t)[10], n_active_local_gpu[N_GPU], n_active_local_max,n_active_local_min;
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	int timelevel_cutoff = MY_MIN(AMR_MAXTIMELEVEL * (1 + 3 * RAD_M1 * DYNAMIC_LOADBALANCE), 32 * (1 + 3 * RAD_M1 * DYNAMIC_LOADBALANCE));
	int numtasks_local = numtasks*N_GPU;
	int min_steps, max_steps, total_steps;
	double val;
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	#if(RAD_M1)
	double max_weight = 1;
	#else
	double max_weight = MAX_WEIGHT;
	#endif
	//rm_order2();
	n_ord_total_RM_t=(int(*)[10])calloc(NB, sizeof(int[10]));

	if (numtasks_local > NB && rank == 0) fprintf(stderr, "Warning: numtasks_local is smaller than NB. Watch out for crashes! \n");


	#if(DYNAMIC_LOADBALANCE)
	if (nstep>0) {
		double max_time = 0.0;
		double min_time = 1000000000000.0;
		double avg_time = 0.0;
		MPI_Barrier(MPI_COMM_WORLD);
	
		//First benchmark blocks on node
		if (rank == 0) fprintf(stderr, "Starting benchmarking step! \n");
		for (n = 0; n < n_active; n++)benchmark_GPU(n_ord[n]);

		//Synchronize with other nodes
		for (n = 0; n < n_active_total; n++) rc = MPI_Ibcast(&(bench_time[n_ord_total[n]]), 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], mpi_cartcomm, &request_timelevel[n_ord_total[n]]);

		//Receive from other nodes using blocking receive and calculate maximum time
		for (n = 0; n < n_active_total; n++) {
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[0][0]);
			max_time = MY_MAX(max_time, bench_time[n_ord_total[n]]);
			min_time = MY_MIN(min_time, bench_time[n_ord_total[n]]);
			avg_time += bench_time[n_ord_total[n]];
		}
		avg_time /= (double)n_active_total;

		if (rank == 0)  fprintf(stderr, "max_time: %f, min_time: %f, avg_time: %f \n", max_time, min_time, avg_time);
		for (n = 0; n < n_active_total; n++) {
			block[n_ord_total[n]][AMR_WEIGHT] = 4;
			//if (bench_time[n_ord_total[n]] > 0.34 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 16;
			if (bench_time[n_ord_total[n]] > 0.67 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 2;
			if (bench_time[n_ord_total[n]] > 1.33 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 1;
			//if (bench_time[n_ord_total[n]] > 2.66 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 2;
			//if (bench_time[n_ord_total[n]] > 5.32 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 1;
			max_weight = MY_MAX(max_weight, block[n_ord_total[n]][AMR_WEIGHT]);
		}

		/*for (n = 0; n < n_active_total; n++) {
			block[n_ord_total[n]][AMR_WEIGHT] = 4;
			//if (bench_time[n_ord_total[n]] > 0.34 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 16;
			if (bench_time[n_ord_total[n]] > 0.5 * max_time)block[n_ord_total[n]][AMR_WEIGHT] = 2;
			if (bench_time[n_ord_total[n]] > 1.33 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 1;
			//if (bench_time[n_ord_total[n]] > 2.66 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 2;
			//if (bench_time[n_ord_total[n]] > 5.32 * avg_time)block[n_ord_total[n]][AMR_WEIGHT] = 1;
			max_weight = MY_MAX(max_weight, block[n_ord_total[n]][AMR_WEIGHT]);
		}*/

		//for (n = 0; n < n_active_total; n++) {
		//	block[n_ord_total[n]][AMR_WEIGHT] = (int)pow(2, round(log(max_time / bench_time[n_ord_total[n]]) / log(2)));
		//	max_weight = MY_MAX(max_weight, block[n_ord_total[n]][AMR_WEIGHT]);
		//	//if (rank == 0)  fprintf(stderr, "AMR_WEIGHT: %d \n", block[n_ord_total[n]][AMR_WEIGHT]);
		//}

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) fprintf(stderr, "Benchmarking step finished! \n");
	}
	else {
		for (n = 0; n < n_active_total; n++) block[n_ord_total[n]][AMR_WEIGHT] = 1;
	}
	#endif
	
	do{
		count++;
		//First make a z-order curve for each timelevel seperately, then load balance for timesteps. This is the best and most advanced method
		for (i = 0; i <= round(log(timelevel_cutoff) / log(2)); i++){
			n_active_total_t[i] = 0;
			n_active_total_steps_t[i] = 0;
		}
		int tl;
		//Order active blocks in an ordered array and keep track of the number of blocks and timesteps at each timelevel
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total_RM[n]][AMR_ACTIVE] == 1){
				tl = MY_MIN(round(log(block[n_ord_total_RM[n]][AMR_TIMELEVEL] * block[n_ord_total_RM[n]][AMR_WEIGHT]) / log(2.0)), log(timelevel_cutoff) / log(2.0));
				n_ord_total_RM_t[n_active_total_t[tl]][tl] = n_ord_total_RM[n];
				n_active_total_steps_t[tl] += timelevel_cutoff / MY_MIN(block[n_ord_total_RM[n]][AMR_TIMELEVEL] * block[n_ord_total_RM[n]][AMR_WEIGHT], timelevel_cutoff);
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
				val=MY_MIN(block[n_ord_total_RM_t[0][i]][AMR_TIMELEVEL] * block[n_ord_total_RM_t[0][i]][AMR_WEIGHT], timelevel_cutoff);
				nr_timesteps = timelevel_cutoff / val;
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
				val = MY_MIN(block[n_ord_total_RM_t[n][i]][AMR_TIMELEVEL] * block[n_ord_total_RM_t[n][i]][AMR_WEIGHT], timelevel_cutoff);
				nr_timesteps = timelevel_cutoff / val;
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
		//#pragma omp parallel for schedule(dynamic,1) private(n, temp)
		for (n = 0; n < n_active_total; n++){
			temp = NODE[n_ord_total_RM[n]];
			NODE[n_ord_total_RM[n]] = temp / N_GPU;
			GPU[n_ord_total_RM[n]] = (temp - NODE[n_ord_total_RM[n]] * N_GPU);
			if (rank == NODE[n_ord_total_RM[n]]) {
				//#pragma omp critical
				//{
					n_active_local_gpu[GPU[n_ord_total_RM[n]]]++;
				//}
			}
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

		if ((n_active_local_max> max_blocks/numtasks || n_active_local_min < 1) && timelevel_cutoff >= 2) timelevel_cutoff /= 2;
	} while ((n_active_local_max> max_blocks/numtasks || n_active_local_min < 1) && count < round(log(AMR_MAXTIMELEVEL) / log(2)) + 1);
	
	if (n_active_local_max > max_blocks/numtasks) {
		if(rank==0)fprintf(stderr, "Error in balance_load: Too many blocks present, increase MAX_BLOCKS if you have enough (GPU)RAM! \n");
		//exit(0);
	}
	if (rank == 0) fprintf(stderr, "Load balance started with cutoff balancing level %d! \n", timelevel_cutoff);
	
	//#pragma omp parallel for schedule(dynamic,1) private(n, rc)
	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total_RM[n]][AMR_NODE] != NODE[n_ord_total_RM[n]]){
			if (block[n_ord_total_RM[n]][AMR_NODE] == rank){
				rc = MPI_Isend(&p[nl[n_ord_total_RM[n]]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[n]], (5 * NB_LOCAL + block[n_ord_total_RM[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[n]]][0]);
				#if STAGGERED
				rc += MPI_Isend(&ps[nl[n_ord_total_RM[n]]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[n_ord_total_RM[n]], (6 * NB_LOCAL + block[n_ord_total_RM[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_ord_total_RM[n]]][1]);
				#endif
				if (rc != 0)fprintf(stderr, "Error balance_load send %d", rc);
			}

			if (NODE[n_ord_total_RM[n]] == rank){
				//Allocate memory for active blocks on node
				temp_p[n_ord_total_RM[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
				temp_ps[n_ord_total_RM[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
				if (block[n_ord_total_RM[n]][AMR_NODE] >= 0){
					rc = MPI_Irecv(&temp_p[n_ord_total_RM[n]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[n]][AMR_NODE], (5 * NB_LOCAL + block[n_ord_total_RM[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp1[n_ord_total_RM[n]]);
					#if STAGGERED
					rc += MPI_Irecv(&temp_ps[n_ord_total_RM[n]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[n]][AMR_NODE], (6 * NB_LOCAL + block[n_ord_total_RM[n]][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqstemp2[n_ord_total_RM[n]]);
					#endif
					if (rc != 0)fprintf(stderr, "Error balance_load receive %d", rc);
				}
			}
		}
	}

	//#pragma omp parallel for schedule(dynamic,1) private(n)
	for (n = 0; n < n_active_total; n++){
		//Then use MPI_wait to clean up data that has been sent
		if (block[n_ord_total_RM[n]][AMR_NODE] != NODE[n_ord_total_RM[n]]){
			if (block[n_ord_total_RM[n]][AMR_NODE] == rank){
				//#pragma omp critical
				//{
					MPI_Wait(&boundreqs[nl[n_ord_total_RM[n]]][0], &Statbound[nl[n_ord_total_RM[n]]][0]);
					#if STAGGERED
					MPI_Wait(&boundreqs[nl[n_ord_total_RM[n]]][1], &Statbound[nl[n_ord_total_RM[n]]][1]);
					#endif
					free_arrays(n_ord_total_RM[n]);
					#if(GPU_ENABLED || GPU_DEBUG )
					GPU_finish(n_ord_total_RM[n], 0);
					#endif
				//}
			}
			block[n_ord_total_RM[n]][AMR_GPU] = -1;

			if (NODE[n_ord_total_RM[n]] == rank){
				//#pragma omp critical
				//{
					if (block[n_ord_total_RM[n]][AMR_NODE] >= 0) {
						MPI_Wait(&boundreqstemp1[n_ord_total_RM[n]], &Statbound[0][10]);
						#if STAGGERED
						MPI_Wait(&boundreqstemp2[n_ord_total_RM[n]], &Statbound[0][11]);
						#endif
					}
					set_arrays(n_ord_total_RM[n]);
					set_grid(n_ord_total_RM[n]);
					#pragma omp parallel for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)  private(i, j, z, k)
					ZSLOOP3D(N1_GPU_offset[n_ord_total_RM[n]] - N1G, N1_GPU_offset[n_ord_total_RM[n]] + BS_1 - 1 + N1G, N2_GPU_offset[n_ord_total_RM[n]] - N2G, N2_GPU_offset[n_ord_total_RM[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord_total_RM[n]] - N3G, N3_GPU_offset[n_ord_total_RM[n]] + BS_3 - 1 + N3G) {
						PLOOP p[nl[n_ord_total_RM[n]]][index_3D(n_ord_total_RM[n], i, j, z)][k] = temp_p[n_ord_total_RM[n]][index_3D(n_ord_total_RM[n], i, j, z)][k];
						#if(STAGGERED)
						for (k = 0; k < NDIM; k++) ps[nl[n_ord_total_RM[n]]][index_3D(n_ord_total_RM[n], i, j, z)][k] = temp_ps[n_ord_total_RM[n]][index_3D(n_ord_total_RM[n], i, j, z)][k];
						#endif
					}
					free(temp_p[n_ord_total_RM[n]]);
					free(temp_ps[n_ord_total_RM[n]]);
				//}
			}
		}

		//Set to updated node
		block[n_ord_total_RM[n]][AMR_NODE] = NODE[n_ord_total_RM[n]];
	}

	//Now reloadbalance between the GPUs on a single node
	//#pragma omp parallel for schedule(dynamic,1) private(n)
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total_RM[n]][AMR_NODE] == rank){
			if (GPU[n_ord_total_RM[n]] != block[n_ord_total_RM[n]][AMR_GPU]){
				//#pragma omp critical
				//{
					#if(GPU_ENABLED || GPU_DEBUG )
					set_arrays_GPU(n_ord_total_RM[n], GPU[n_ord_total_RM[n]]);
					GPU_write(n_ord_total_RM[n]);
					#endif
				//}
			}
		}
		block[n_ord_total_RM[n]][AMR_GPU] = GPU[n_ord_total_RM[n]];
	}

	activate_blocks();
	
	//Set timelevel communicator
	set_communicator();

	min_steps = 0; max_steps = 0; total_steps = 0;
	count_node[0]=0;
	for (g = 0; g < N_GPU; g++) count_gpu[g] = 0;
	for (n = 0; n < n_active; n++) {
		count_node[0] += max_weight * AMR_MAXTIMELEVEL / (block[n_ord[n]][AMR_WEIGHT] * block[n_ord[n]][AMR_TIMELEVEL]);
		count_gpu[block[n_ord[n]][AMR_GPU]] += max_weight * AMR_MAXTIMELEVEL / (block[n_ord[n]][AMR_WEIGHT] * block[n_ord[n]][AMR_TIMELEVEL]);
	}
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

	#if(GPU_ENABLED)
	//Calculate memory consumption on each GPU
	double max_mem = 0.;
	double min_mem = 0.;
	double total_mem = 0.;
	double mem = 0.;
	#if(CUDA_MEMCALC)
	size_t mem_int, mem_tot;
	gpuMemGetInfo(&mem_int, &mem_tot);
	mem = (double)(mem_tot - mem_int) / pow(10., 9.);
	max_mem += mem;
	min_mem += mem;
	total_mem += mem;
	#else
	for (n = 0; n < n_active; n++) {
		mem = calc_mem_gpu(n_ord[n]) / pow(10., 9.);
		max_mem += mem;
		min_mem += mem;
		total_mem += mem;
	}
	#endif
	MPI_Allreduce(MPI_IN_PLACE, &min_mem, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &max_mem, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &total_mem, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	double mem_per_block = total_mem / n_active_total;
	#if(CUDA_MEMCALC)
	max_blocks = (int)(n_active_total + (gpu_mem * numtasks - total_mem) / mem_per_block);
	#else
	max_blocks = (int)(n_active_total + (GPU_MEM * numtasks - total_mem) / mem_per_block);
	#endif
	if (rank == 0) fprintf(stderr, "GPU memory consumption in GB (total, min, max): %f %f %f \n", total_mem, min_mem, max_mem);
	if (rank == 0) fprintf(stderr, "Max blocks set to: %d \n", max_blocks);
	#else
	//Calculate memory consumption on each CPU
	double max_mem = 0.;
	double min_mem = 0.;
	double total_mem = 0.;
	double mem = 0.;
	for (n = 0; n < n_active; n++) {
		mem = calc_mem_gpu(n_ord[n]) / pow(10., 9.);
		max_mem += mem;
		min_mem += mem;
		total_mem += mem;
	}
	MPI_Allreduce(MPI_IN_PLACE, &min_mem, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &max_mem, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &total_mem, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	double mem_per_block = total_mem / n_active_total;
	max_blocks = (int)(n_active_total + (CPU_MEM * numtasks - total_mem) / mem_per_block);

	if (rank == 0) fprintf(stderr, "CPU memory consumption in GB (total, min, max): %f %f %f \n", total_mem, min_mem, max_mem);
	if (rank == 0) fprintf(stderr, "Max blocks set to: %d \n", max_blocks);
	#endif

	bound_prim(p, 1);
	//Copy the B-field to make the code resilient against two bit ECC errors
	#if(GPU_ENABLED)
	for (n = 0; n < n_active; n++){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		#pragma omp parallel private(i, j, z, k)
		{
			#pragma omp for collapse(3) schedule(static,((BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G))/nthreads)
			ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, N1_GPU_offset[n_ord[n]] + BS_1 - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G){
				#if(STAGGERED)
				for (k = 1; k < NDIM; k++){
					ps_1[nl[n_ord[n]]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]]) + (i - N1_GPU_offset[n_ord[n]] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n_ord[n]] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n_ord[n]] + N3G)] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];
				}
				#endif
			}
		}
		gpuMemcpyAsync(Bufferps_1[nl[n_ord[n]]], ps_1[nl[n_ord[n]]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n_ord[n]]]);
	}
	//GPU_boundprim(1);
	#endif
	free(n_ord_total_RM_t);

	if (rank == 0) fprintf(stderr, "Load balance finished! \n");
}

void balance_load_gpu(void){
}

/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
void activate_blocks(void){
	int n, i, g;
	int n_ord_gpu[NB_LOCAL], n_active_gpu[N_GPU], n_g[N_GPU];
	n_active = 0;
	n_active_total = 0;

	#if(N_GPU>1)
	for (g = 0; g < N_GPU; g++)n_active_gpu[g] = 0;
	#endif

	for (n = 0; n < MY_MIN(numtasks * N_GPU, NB); n++) NODE_global[n] = 0;
	for (n = 0; n < numtasks; n++) n_active_node[n] = 0;

	for (n = 0; n <= n_max; n++) block[n][AMR_REFINED] = 0;
	for (n = 0; n <= n_max; n++){
		if (block[n][AMR_ACTIVE] == 1 && block[n][AMR_NODE] == rank){
			//Order active blocks into array n_ord and keep track of number of active block in n_active
			n_ord[n_active] = n;
			n_ord_RM[n_active] = n;
			n_active++;	
			#if(N_GPU>1)
			n_ord_gpu[block[n][AMR_GPU]] = n_active_gpu[block[n][AMR_GPU]];
			n_active_gpu[block[n][AMR_GPU]]++;
			#endif
		}
		if (block[n][AMR_ACTIVE] == 1){
			//Order active blocks into array n_ord and keep track of number of active block in n_active_total
			
			n_ord_total[n_active_total] = n;
			n_ord_total_RM[n_active_total] = n;
            if (block[n][AMR_NODE] >= 0){
				n_ord_node[block[n][AMR_NODE]][n_active_node[block[n][AMR_NODE]]] = n;
				block[n][AMR_NUMBER] = NODE_global[block[n][AMR_NODE]];
				NODE_global[block[n][AMR_NODE]]++;
				n_active_node[block[n][AMR_NODE]] = NODE_global[block[n][AMR_NODE]];
            }
			n_active_total++;
			if (block[n][AMR_LEVEL] > 0) block[block[n][AMR_PARENT]][AMR_REFINED] = 1;
		}
	}
	#if(N_GPU>1)
	for (n = 0; n < MY_MIN(numtasks * N_GPU, NB); n++) NODE_global[n] = 0;
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_NODE] >= 0) NODE_global[block[n_ord_total[n]][AMR_NODE]*N_GPU + (block[n_ord_total[n]][AMR_GPU])]++;
	}
	n=0;
	for(g=0;g<N_GPU;g++) n_g[g]=0;
	while(n < n_active){
		for(g=0;g<N_GPU;g++){
			if(n_g[g]<n_active_gpu[g]){
				n_ord[n] = n_ord_gpu[n_g[g]];
				n_g[g]++;
				n++;
			}
		}
	}	
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
}

void block_average(int n, int n_child, int i1, int i2, int j1, int j2, int z1, int z2){
	int i, j, z, k, ic, jc, zc, i_1, i_2, j_1, j_2, z_1, z_2;
	struct of_geom geom;
	struct of_state q;
	int ref_1, ref_2, ref_3;

	ref_1 = block[n_child][AMR_LEVEL1] - block[n][AMR_LEVEL1];
	ref_2 = block[n_child][AMR_LEVEL2] - block[n][AMR_LEVEL2];
	ref_3 = block[n_child][AMR_LEVEL3] - block[n][AMR_LEVEL3];

	#pragma omp parallel private(i, j, z, k, ic, jc, zc, i_1, i_2, j_1, j_2, z_1, z_2,q,geom)
	{
		//Average primitive quantities
		#pragma omp for collapse(3) schedule(static, (i2-i1)*(j2-j1)*(z2-z1)/nthreads)
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
		/*
		#pragma omp for collapse(3) schedule(static, (i2-i1)*(j2-j1)*(z2-z1)/nthreads)
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
		#pragma omp for collapse(3) schedule(static, (i2-i1+D1)*(j2-j1+D2)*(z2-z1+D3)/nthreads)
		for (i = i1; i < i2 + D1; i++){
			for (j = j1; j < j2 + D2; j++){
				for (z = z1; z < z2 + D3; z++){
					ic = (i - i1)*(1 + ref_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + ref_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + ref_3) + N3_GPU_offset[n_child];
					k = 1;
					if (j < j2 && z < z2){
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] * 0.25*(
							ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE1] +
							ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc)][FACE1] +
							ps[nl[n_child]][index_3D(n_child, ic, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + ref_3)][FACE1] +
							ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc + ref_3)][FACE1]);
					}
					k = 2;
					if (i < i2 && z < z2){
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] * 0.25*(
							ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE2] +
							ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc)][FACE2] +
							ps[nl[n_child]][index_3D(n_child, ic, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc + ref_3)][FACE2] +
							ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc + ref_3)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc + ref_3)][FACE2]);
					}
					k = 3;
					if (j < j2 && i < i2){
						ps[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 1. / gdet[nl[n]][index_2D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] * 0.25*(
							ps[nl[n_child]][index_3D(n_child, ic, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc, zc)][FACE3] +
							ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc, zc)][FACE3] +
							ps[nl[n_child]][index_3D(n_child, ic, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic, jc + ref_2, zc)][FACE3] +
							ps[nl[n_child]][index_3D(n_child, ic + ref_1, jc + ref_2, zc)][k] * gdet[nl[n_child]][index_2D(n_child, ic + ref_1, jc + ref_2, zc)][FACE3]);
					}
				}
			}
		}
		#endif
	}
}

void derefine(int n){
	int i,j,z,k, n_child;
	int ref_1, ref_2, ref_3;
	if (rank == 0 && numtasks<400) fprintf(stderr, "Derefining block %d %d %d %d \n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
	if (block[n][AMR_ACTIVE] != 0) fprintf(stderr, "Error: Trying to derefine active block %d! \n", n);

	block[n][AMR_ACTIVE] = 1;
	block[n][AMR_GPU] = block[block[n][AMR_CHILD1]][AMR_GPU];
	if (block[n][AMR_NODE] == rank){
		set_arrays(n);
		set_grid(n);

		ref_1 = block[block[n][AMR_CHILD1]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD1]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD1]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD1] >= 0){
			n_child = block[n][AMR_CHILD1];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD1]);
			GPU_finish(block[n][AMR_CHILD1], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD2]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD2]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD2]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD2] >= 0 && ref_3 == 1){
			n_child = block[n][AMR_CHILD2];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD2]);
			GPU_finish(block[n][AMR_CHILD2], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD3]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD3]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD3]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD3] >= 0 && ref_2 == 1){
			n_child = block[n][AMR_CHILD3];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD3]);
			GPU_finish(block[n][AMR_CHILD3], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD4]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD4]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD4]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD4] >= 0 && ref_2 == 1 && ref_3 == 1){
			n_child = block[n][AMR_CHILD4];
			block_average(n, n_child, 0, BS_1 / (1 + ref_1), BS_2 / (1 + ref_2), BS_2, BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD4]);
			GPU_finish(block[n][AMR_CHILD4], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD5]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD5]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD5]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD5] >= 0 && ref_1 == 1){
			n_child = block[n][AMR_CHILD5];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD5]);
			GPU_finish(block[n][AMR_CHILD5], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD6]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD6]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD6]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD6] >= 0 && ref_1 == 1 && ref_3 == 1){
			n_child = block[n][AMR_CHILD6];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, 0, BS_2 / (1 + ref_2), BS_3 / (1 + ref_3), BS_3);
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD6]);
			GPU_finish(block[n][AMR_CHILD6], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD7]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD7]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD7]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_CHILD7] >= 0 && ref_1 == 1 && ref_2 == 1){
			n_child = block[n][AMR_CHILD7];
			block_average(n, n_child, BS_1 / (1 + ref_1), BS_1, BS_2 / (1 + ref_2), BS_2, 0, BS_3 / (1 + ref_3));
			#if(GPU_ENABLED || GPU_DEBUG )
			free_arrays(block[n][AMR_CHILD7]);
			GPU_finish(block[n][AMR_CHILD7], 0);
			#endif
		}

		ref_1 = block[block[n][AMR_CHILD8]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD8]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		ref_3 = block[block[n][AMR_CHILD8]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
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
	block[n][AMR_TIMELEVEL] = MY_MIN(AMR_MAXTIMELEVEL, 2*min_timelevel);
	//if (block[n][AMR_POLE] >= 1)block[n][AMR_TIMELEVEL] = 1;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_ACTIVE] = 0;
	for (i = AMR_CHILD1; i <= AMR_CHILD8; i++)block[block[n][i]][AMR_TIMELEVEL] = 1;

	#if(GPU_ENABLED || GPU_DEBUG )
	if (block[n][AMR_NODE] == rank){
		if (block[n][AMR_GPU] == -1 && GPU_ENABLED) fprintf(stderr, "Only positive values allowed for device number! \n");
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
		#pragma omp for collapse(3) schedule(static, BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(0, BS_1 - 1, 0, BS_2 - 1, 0, BS_3 - 1) {
			i1 = (i - i % (1 + ref_1)) / (1 + ref_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * ref_1;
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * ref_2;
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * ref_3;
			PLOOP{
				prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
				prim[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.5 * (-0.5 + i % (1 + ref_1)) * ref_1 * d1[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.5 * (-0.5 + j % (1 + ref_2)) * ref_2 * d2[nl[n]][index_3D(n, i1, j1, z1)][k] + 0.5 * (-0.5 + z % (1 + ref_3)) * ref_3 * d3[nl[n]][index_3D(n, i1, j1, z1)][k];
			}
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][RHO] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][RHO]);
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][UU] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][UU]);		


			#if(DO_YE)
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][YE] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][YE]);
			#endif
			#if(TWO_T)
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][ENTRE] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][ENTRE]);
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][ENTRI] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][ENTRI]);
			#endif
			#if(RAD_M1)
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][UU_RAD] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][UU_RAD]);
			#endif
			#if(P_NUM)
			prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][PHOTON] = fabs(prim[nl[n_child]][index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][PHOTON]);
			#endif

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
	int ref_1=0, ref_2=0, ref_3=0;
	double *pointer1, *pointer2, *pointer3, *pointer4, *pointer5, *pointer6;

	//Use divergence free prolongation to handle boundaries
	n_rec1 = -1;
	n_rec2 = -1;
	n_rec3 = -1;
	n_rec4 = -1;
	n_rec5 = -1;
	n_rec6 = -1;
	if (block[n][AMR_NBR2_1] >= 0) set_ref(n, block[n][AMR_NBR2_1], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 0 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR2_1]][AMR_PARENT]][AMR_REFINED] == 1){ pointer4 = receive4_5[nl[n]]; n_rec4 = AMR_NBR2_1; }
	if (block[n][AMR_NBR2_2] >= 0) set_ref(n, block[n][AMR_NBR2_2], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 0 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_2] >= 0 && block[block[n][AMR_NBR2_2]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR2_2]][AMR_PARENT]][AMR_REFINED] == 1){ pointer4 = receive4_6[nl[n]]; n_rec4 = AMR_NBR2_2; }
	if (block[n][AMR_NBR2_3] >= 0) set_ref(n, block[n][AMR_NBR2_3], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 1 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_3] >= 0 && block[block[n][AMR_NBR2_3]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR2_3]][AMR_PARENT]][AMR_REFINED] == 1){ pointer4 = receive4_7[nl[n]]; n_rec4 = AMR_NBR2_3; }
	if (block[n][AMR_NBR2_4] >= 0) set_ref(n, block[n][AMR_NBR2_4], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 1 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR2] >= 0 && block[n][AMR_NBR2_4] >= 0 && block[block[n][AMR_NBR2_4]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR2_4]][AMR_PARENT]][AMR_REFINED] == 1){ pointer4 = receive4_8[nl[n]]; n_rec4 = AMR_NBR2_4; }

	if (block[n][AMR_NBR4_5] >= 0) set_ref(n, block[n][AMR_NBR4_5], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 0 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR4_5]][AMR_PARENT]][AMR_REFINED] == 1){ pointer2 = receive2_1[nl[n]]; n_rec2 = AMR_NBR4_5; }
	if (block[n][AMR_NBR4_6] >= 0) set_ref(n, block[n][AMR_NBR4_6], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 0 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_6] >= 0 && block[block[n][AMR_NBR4_6]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR4_6]][AMR_PARENT]][AMR_REFINED] == 1){ pointer2 = receive2_2[nl[n]]; n_rec2 = AMR_NBR4_6; }
	if (block[n][AMR_NBR4_7] >= 0) set_ref(n, block[n][AMR_NBR4_7], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 1 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_7] >= 0 && block[block[n][AMR_NBR4_7]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR4_7]][AMR_PARENT]][AMR_REFINED] == 1){ pointer2 = receive2_3[nl[n]]; n_rec2 = AMR_NBR4_7; }
	if (block[n][AMR_NBR4_8] >= 0) set_ref(n, block[n][AMR_NBR4_8], &ref_1, &ref_2, &ref_3);
	if (offset_2 / (2 - ref_2) == 1 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR4] >= 0 && block[n][AMR_NBR4_8] >= 0 && block[block[n][AMR_NBR4_8]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR4_8]][AMR_PARENT]][AMR_REFINED] == 1){ pointer2 = receive2_4[nl[n]]; n_rec2 = AMR_NBR4_8; }

	if (block[n][AMR_NBR1_3] >= 0) set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
	if (offset_1 / (2 - ref_1) == 0 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR1_3]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_1[nl[n]]; n_rec3 = AMR_NBR1_3; }
	if (offset_1 / (2 - ref_1) == 0 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_4] >= 0 && block[block[n][AMR_NBR1_4]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR1_4]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_2[nl[n]]; n_rec3 = AMR_NBR1_4; }
	if (offset_1 / (2 - ref_1) == 1 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_7] >= 0 && block[block[n][AMR_NBR1_7]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR1_7]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_5[nl[n]]; n_rec3 = AMR_NBR1_7; }
	if (offset_1 / (2 - ref_1) == 1 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR1] >= 0 && block[n][AMR_NBR1_8] >= 0 && block[block[n][AMR_NBR1_8]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR1_8]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 2)){ pointer3 = receive3_6[nl[n]]; n_rec3 = AMR_NBR1_8; }

	if (block[n][AMR_NBR3_1] >= 0) set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
	if (offset_1 / (2 - ref_1) == 0 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR3_1]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_3[nl[n]]; n_rec1 = AMR_NBR3_1; }
	if (offset_1 / (2 - ref_1) == 0 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_2] >= 0 && block[block[n][AMR_NBR3_2]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR3_2]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_4[nl[n]]; n_rec1 = AMR_NBR3_2; }
	if (offset_1 / (2 - ref_1) == 1 && offset_3 / (2 - ref_3) == 0 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_5] >= 0 && block[block[n][AMR_NBR3_5]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR3_5]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_7[nl[n]]; n_rec1 = AMR_NBR3_5; }
	if (offset_1 / (2 - ref_1) == 1 && offset_3 / (2 - ref_3) == 1 && block[n][AMR_NBR3] >= 0 && block[n][AMR_NBR3_6] >= 0 && block[block[n][AMR_NBR3_6]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR3_6]][AMR_PARENT]][AMR_REFINED] == 1 && (block[n][AMR_POLE] == 0 || block[n][AMR_POLE] == 1)){ pointer1 = receive1_8[nl[n]]; n_rec1 = AMR_NBR3_6; }

	if (block[n][AMR_NBR6_2] >= 0) set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
	if (offset_1 / (2 - ref_1) == 0 && offset_2 / (2 - ref_2) == 0 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR6_2]][AMR_PARENT]][AMR_REFINED] == 1){ pointer5 = receive5_1[nl[n]]; n_rec5 = AMR_NBR6_2; }
	if (offset_1 / (2 - ref_1) == 0 && offset_2 / (2 - ref_2) == 1 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_4] >= 0 && block[block[n][AMR_NBR6_4]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR6_4]][AMR_PARENT]][AMR_REFINED] == 1){ pointer5 = receive5_3[nl[n]]; n_rec5 = AMR_NBR6_4; }
	if (offset_1 / (2 - ref_1) == 1 && offset_2 / (2 - ref_2) == 0 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_6] >= 0 && block[block[n][AMR_NBR6_6]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR6_6]][AMR_PARENT]][AMR_REFINED] == 1){ pointer5 = receive5_5[nl[n]]; n_rec5 = AMR_NBR6_6; }
	if (offset_1 / (2 - ref_1) == 1 && offset_2 / (2 - ref_2) == 1 && block[n][AMR_NBR6] >= 0 && block[n][AMR_NBR6_8] >= 0 && block[block[n][AMR_NBR6_8]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR6_8]][AMR_PARENT]][AMR_REFINED] == 1){ pointer5 = receive5_7[nl[n]]; n_rec5 = AMR_NBR6_8; }

	if (block[n][AMR_NBR5_1] >= 0) set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
	if (offset_1 / (2 - ref_1) == 0 && offset_2 / (2 - ref_2) == 0 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR5_1]][AMR_PARENT]][AMR_REFINED] == 1){ pointer6 = receive6_2[nl[n]]; n_rec6 = AMR_NBR5_1; }
	if (offset_1 / (2 - ref_1) == 0 && offset_2 / (2 - ref_2) == 1 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_3] >= 0 && block[block[n][AMR_NBR5_3]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR5_3]][AMR_PARENT]][AMR_REFINED] == 1){ pointer6 = receive6_4[nl[n]]; n_rec6 = AMR_NBR5_3; }
	if (offset_1 / (2 - ref_1) == 1 && offset_2 / (2 - ref_2) == 0 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_5] >= 0 && block[block[n][AMR_NBR5_5]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR5_5]][AMR_PARENT]][AMR_REFINED] == 1){ pointer6 = receive6_6[nl[n]]; n_rec6 = AMR_NBR5_5; }
	if (offset_1 / (2 - ref_1) == 1 && offset_2 / (2 - ref_2) == 1 && block[n][AMR_NBR5] >= 0 && block[n][AMR_NBR5_7] >= 0 && block[block[n][AMR_NBR5_7]][AMR_ACTIVE] == 1 && block[block[block[n][AMR_NBR5_7]][AMR_PARENT]][AMR_REFINED] == 1){ pointer6 = receive6_8[nl[n]]; n_rec6 = AMR_NBR5_7; }

	isize = BS_1;
	jsize = BS_2;
	zsize = BS_3;
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, ind0, ind2,b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8,b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8,set_1, set_2,set_3,set_4,set_5,set_6, ref_1, ref_2, ref_3)
	{
		#pragma omp for collapse(3) schedule(static, (BS_1+D1)*(BS_2+D2)*(BS_3+D3)/nthreads)
		ZSLOOP3D(0, BS_1 - 1 + D1, 0, BS_2 - 1 + D2, 0, BS_3 - 1 + D3) {
			//index of child
			ind0 = index_3D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);
			ind2 = index_2D(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);

			//face centered magnetic field components of neighbouring cells that are refined allready
			b1_1 = b1_2 = b1_3 = b1_4 = b1_5 = b1_6 = b1_7 = b1_8 = 0.;
			b2_1 = b2_2 = b2_3 = b2_4 = b2_5 = b2_6 = b2_7 = b2_8 = 0.;
			b3_1 = b3_2 = b3_3 = b3_4 = b3_5 = b3_6 = b3_7 = b3_8 = 0.;
			set_1 = set_2 = set_3 = set_4 = set_5 = set_6 = -1;

			//transform index of child (i,j,z) to index of parent block (i1,j1,z1)
			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			i1 = (i - i % (1 + ref_1)) / (1 + ref_1);
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2);
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3);

			if ((i == 0 || i == ref_1) && offset_1 == 0 && n_rec2 >= 0){
				set_ref(n, block[n][n_rec2], &ref_1, &ref_2, &ref_3);
				if (offset_2 == 1 && ref_2 == 0) j1 += BS_2 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 += BS_3 / 2;
				b1_1 = pointer2[j1*(1 + ref_2)*zsize + z1*(1 + ref_3)];
				b1_2 = pointer2[j1*(1 + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				b1_3 = pointer2[(j1*(1 + ref_2) + ref_2)*zsize + z1*(1 + ref_3)];
				b1_4 = pointer2[(j1*(1 + ref_2) + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				if (offset_2 == 1 && ref_2 == 0) j1 -= BS_2 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 -= BS_3 / 2;
				set_2 = 1;
			}
			else set_2 = -1;
			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			if ((i == BS_1 || i == BS_1 - ref_1 || i == BS_1 - (1 + ref_1)) && (offset_1 == 1 || ref_1 == 0) && n_rec4 >= 0){
				set_ref(n, block[n][n_rec4], &ref_1, &ref_2, &ref_3);
				if (offset_2 == 1 && ref_2 == 0) j1 += BS_2 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 += BS_3 / 2;
				b1_5 = pointer4[j1*(1 + ref_2)*zsize + z1*(1 + ref_3)];
				b1_6 = pointer4[j1*(1 + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				b1_7 = pointer4[(j1*(1 + ref_2) + ref_2)*zsize + z1*(1 + ref_3)];
				b1_8 = pointer4[(j1*(1 + ref_2) + ref_2)*zsize + (z1*(1 + ref_3) + ref_3)];
				if (offset_2 == 1 && ref_2 == 0) j1 -= BS_2 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 -= BS_3 / 2;
				set_4 = 1;
			}
			else set_4 = -1;

			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			if ((j == 0 || j == ref_2) && offset_2 == 0 && n_rec3 >= 0) {
				set_ref(n, block[n][AMR_NBR1_3], &ref_1, &ref_2, &ref_3);
				if (offset_1 == 1 && ref_1 == 0) i1 += BS_1 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 += BS_3 / 2;
				b2_1 = pointer3[i1*(1 + ref_1)*zsize + z1*(1 + ref_3)];
				b2_2 = pointer3[i1*(1 + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				b2_5 = pointer3[(i1*(1 + ref_1) + ref_1)*zsize + z1*(1 + ref_3)];
				b2_6 = pointer3[(i1*(1 + ref_1) + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				if (offset_1 == 1 && ref_1 == 0) i1 -= BS_1 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 -= BS_3 / 2;
				set_3 = 1;
			}
			else set_3 = -1;
			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			if ((j == BS_2 || j == BS_2 - ref_2 || j == BS_2 - (1 + ref_2)) && (offset_2 == 1 || ref_2 == 0) && n_rec1 >= 0) {
				set_ref(n, block[n][AMR_NBR3_1], &ref_1, &ref_2, &ref_3);
				if (offset_1 == 1 && ref_1 == 0) i1 += BS_1 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 += BS_3 / 2;
				b2_3 = pointer1[i1*(1 + ref_1)*zsize + z1*(1 + ref_3)];
				b2_4 = pointer1[i1*(1 + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				b2_7 = pointer1[(i1*(1 + ref_1) + ref_1)*zsize + z1*(1 + ref_3)];
				b2_8 = pointer1[(i1*(1 + ref_1) + ref_1)*zsize + (z1*(1 + ref_3) + ref_3)];
				if (offset_1 == 1 && ref_1 == 0) i1 -= BS_1 / 2;
				if (offset_3 == 1 && ref_3 == 0) z1 -= BS_3 / 2;
				set_1 = 1;
			}
			else set_1 = -1;

			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			if ((z == 0 || z == ref_3) && offset_3 == 0 && n_rec5 >= 0) {
				set_ref(n, block[n][AMR_NBR6_2], &ref_1, &ref_2, &ref_3);
				if (offset_1 == 1 && ref_1 == 0) i1 += BS_1 / 2;
				if (offset_2 == 1 && ref_2 == 0) j1 += BS_2 / 2;
				b3_1 = pointer5[i1*(1 + ref_1)*jsize + j1*(1 + ref_2)];
				b3_3 = pointer5[i1*(1 + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				b3_5 = pointer5[(i1*(1 + ref_1) + ref_1)*jsize + j1*(1 + ref_2)];
				b3_7 = pointer5[(i1*(1 + ref_1) + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				if (offset_1 == 1 && ref_1 == 0) i1 -= BS_1 / 2;
				if (offset_2 == 1 && ref_2 == 0) j1 -= BS_2 / 2;
				set_5 = 1;
			}
			else set_5 = -1;
			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			if ((z == BS_3 || z == BS_3 - ref_3 || z == BS_3 - (1 + ref_3)) && (offset_3 == 1 || ref_3 == 0) && n_rec6 >= 0) {
				set_ref(n, block[n][AMR_NBR5_1], &ref_1, &ref_2, &ref_3);
				if (offset_1 == 1 && ref_1 == 0) i1 += BS_1 / 2;
				if (offset_2 == 1 && ref_2 == 0) j1 += BS_2 / 2;
				b3_2 = pointer6[i1*(1 + ref_1)*jsize + j1*(1 + ref_2)];
				b3_4 = pointer6[i1*(1 + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				b3_6 = pointer6[(i1*(1 + ref_1) + ref_1)*jsize + j1*(1 + ref_2)];
				b3_8 = pointer6[(i1*(1 + ref_1) + ref_1)*jsize + (j1*(1 + ref_2) + ref_2)];
				if (offset_1 == 1 && ref_1 == 0) i1 -= BS_1 / 2;
				if (offset_2 == 1 && ref_2 == 0) j1 -= BS_2 / 2;
				set_6 = 1;
			}
			else set_6 = -1;
			
			set_ref(n, n_child, &ref_1, &ref_2, &ref_3);
			i1 = (i - i % (1 + ref_1)) / (1 + ref_1) + N1_GPU_offset[n] + offset_1*BS_1 / 2 * ref_1 - (i == BS_1 && (offset_1 == 1 || ref_1 == 0));
			j1 = (j - j % (1 + ref_2)) / (1 + ref_2) + N2_GPU_offset[n] + offset_2*BS_2 / 2 * ref_2 - (j == BS_2 && (offset_2 == 1 || ref_2 == 0));
			z1 = (z - z % (1 + ref_3)) / (1 + ref_3) + N3_GPU_offset[n] + offset_3*BS_3 / 2 * ref_3 - (z == BS_3 && (offset_3 == 1 || ref_3 == 0));

			pb[nl[n_child]][ind0][1] =
				1. / gdet[nl[n_child]][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + 0.5*(i % (1 + ref_1)) + (double)(i == BS_1 && (i % (1 + ref_1)!=1) && (offset_1 == 1 || ref_1 == 0)), 0.5*(-0.5 + j % (1 + ref_2)) * ref_2, 0.5*(-0.5 + z % (1 + ref_3)) * ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
				b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			pb[nl[n_child]][ind0][2] =
				1. / gdet[nl[n_child]][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.5*(-0.5 + i % (1 + ref_1)) * ref_1, -0.5 + 0.5*(j % (1 + ref_2)) + (double)(j == BS_2 && (j % (1 + ref_2) != 1) && (offset_2 == 1 || ref_2 == 0)), 0.5*(-0.5 + z % (1 + ref_3)) * ref_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
				b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			pb[nl[n_child]][ind0][3] =
				1. / gdet[nl[n_child]][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.5*(-0.5 + i % (1 + ref_1)) * ref_1, 0.5*(-0.5 + j % (1 + ref_2)) * ref_2, -0.5 + 0.5*(z % (1 + ref_3)) + (double)(z == BS_3 && (z % (1 + ref_3) != 1) && (offset_3 == 1 || ref_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
				b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
		}
	}
}

void pre_refine(void){
	int n1, i, j, z;
	for (n1 = 0; n1 < n_active; n1++) B_send1(ps, Bufferps_1, n_ord[n1]);
	for (n1 = 0; n1 < n_active; n1++) B_rec1(ps, Bufferps_1, n_ord[n1]);
	prolong_grid();

	for (n1 = 0; n1 < n_active; n1++){
		#pragma omp parallel private(i, j, z)
		{
			#pragma omp for collapse(3) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)
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

	if (!check_nesting(n) || NODE_global[block[n][AMR_NODE]*N_GPU + block[n][AMR_GPU]] > max_blocks/numtasks){
		if (rank == 0) fprintf(stderr, "Failed to refine block %d %d %d %d due to memory size on node %d!\n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3], block[n][AMR_NODE]);
		return 0; //First make sure nesting criteria are satisfied
	}
	else{
		if (rank == 0) {
			if (numtasks < 10 || block[n][AMR_COORD1] == 0) {
				fprintf(stderr, "Refining block %d %d %d %d on node %d\n", block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3], block[n][AMR_NODE]);
			}
		}
		ref_1 = block[block[n][AMR_CHILD2]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		ref_2 = block[block[n][AMR_CHILD2]][AMR_LEVEL2] - block[n][AMR_LEVEL2];
		NODE_global[block[n][AMR_NODE] * N_GPU + block[n][AMR_GPU]] += (1 + ref_1)*(1 + ref_2)*(1 + REF_3) - 1;
	}

	if (block[n][AMR_ACTIVE] == 0 && rank == 0) fprintf(stderr, "Watch out: trying to refine non-active block %d \n", n);
	if (block[n][AMR_ACTIVE] == 1){
		gpu_local = block[n][AMR_GPU];
		if (block[n][AMR_NODE] == rank){
			//Calculate gradients, store in flux array F1, F2, F3
			#pragma omp parallel private(i, j, z,k)
			{
				#pragma omp for collapse(3) schedule(static, (BS_1+2*D1)*(BS_2+2*D2)*(BS_3+2*D3)/nthreads)
				ZSLOOP3D(-D1, BS_1 - 1 + D1, -D2, BS_2 - 1 + D2, -D3, BS_3 - 1 + D3) {
					PLOOP{
						F1[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n] - D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n] + D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
						F2[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] - D2, z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + D2, z + N3_GPU_offset[n])][k]);
						#if(N3>1)
						F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] - D3)][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + D3)][k]);
						#else	
						F3[nl[n]][index_3D(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = 0.0;
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

		//Take note that block becomes refined
		for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
			block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL];
			if (block[n][AMR_TIMELEVEL] >= 2)block[block[n][i]][AMR_TIMELEVEL] = block[n][AMR_TIMELEVEL] / 2;
			else reduce_timestep = 1;
			block[block[n][i]][AMR_NODE] = block[n][AMR_NODE];
			block[block[n][i]][AMR_GPU] = gpu_local;
			block[block[n][i]][AMR_ACTIVE] = 1;
		}
		block[n][AMR_GPU] = -1;
		block[n][AMR_ACTIVE] = 0;
		block[n][AMR_TIMELEVEL] = 1;
	}
	return 1;
}

void post_refine(void){
	//Allocate memory for all active blocks
	activate_blocks();
	//set_corners(1);
	#if(N_LEVELS_1D_INT>=0)
	average_grid();
	#endif

	//Set boundary conditions
	bound_prim(p, 1);
	#if(GPU_ENABLED || GPU_DEBUG)
	//GPU_boundprim(1);
	MPI_Barrier(mpi_cartcomm);
	#endif
}

//Checks if neighbouring blocks are sufficiently refined so that no double jumps in refinement level are created
int check_nesting(int n){
	int i,z;
	int flag = 1;
	for (i = AMR_NBR1P; i <= AMR_CORN12P; i++){
		if (block[n][i] >= 0 && block[block[n][i]][AMR_ACTIVE] == 1){
			if (!refine(block[n][i])){
				flag = 0;
			}
		}
	}	

	//Refine around pole
	if (block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 && flag == 1){
		block[n][AMR_TAG] = 1;
		if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1 && block[block[n][AMR_NBR5]][AMR_TAG]!=1){
			block[block[n][AMR_NBR5]][AMR_TAG] = 1;
			if (!refine(block[n][AMR_NBR5])){
				flag = 0;
			}
		}
	}

	return flag;
}

#if WHICHPROBLEM==DISRUPTION_PROBLEM
#define REFINEMENT_CUTOFF 0.0000001
#elif (WHICHPROBLEM==POSTMERGER_PROBLEM)
#define REFINEMENT_CUTOFF 0.02 //in this case density in code units, used for H/R=0.03 disk
#else
#define REFINEMENT_CUTOFF 20000000.2 //in this case density in code units, used for H/R=0.03 disk
#endif

//Refine on basis of some criteria ref_val (not necessary to use rho though, can also be something different)
void check_refcrit(void){
	int n, task, i,j,z,k, l, level, number, ref_1, ref_2, ref_3, i1, i2, i3, tag2;
	int node, n_send, gpu_choice, gpu_counter, var;
	double  rho_rec;
	double(*temp_ps[NB])[NDIM];
	double(*temp_p[NB])[NPR];
	MPI_Request boundreqstemp1[NB], boundreqstemp2[NB];
	if (max_levels == 0) max_levels = N_LEVELS_3D;
	int tag, count=0, begin1, end1;
	int one_block_refined = 0, one_block_derefined=0;
	
	//Start timer
	MPI_Barrier(mpi_cartcomm);
	if (rank == 0) fprintf(stderr, "Starting refinement! \n");
	begin1 = time(NULL);

	//First close dump files in progress
	close_dump();

	#if(DUMP_SMALL)
	close_dump_reduced();
	close_gdump_reduced();
	#endif
	close_gdump();
	close_rdump();

	//Remove boundaries from GPU
	#if(GPU_ENABLED || GPU_DEBUG )
	for(n=0;n<n_active;n++) free_bound_gpu(n_ord[n]);
	#endif

	do{
		count++;
		tag = 0;
		one_block_refined = 0;

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
			#if(GRID_CARTESIAN)
			max_levels = calc_nlevels_tot(block[n_ord_total[n]][AMR_LEVEL], block[n_ord_total[n]][AMR_COORD1], block[n_ord_total[n]][AMR_COORD2], block[n_ord_total[n]][AMR_COORD3]);
			#endif
			if ((ref_val[n_ord_total[n]] > REFINEMENT_CUTOFF || block[n_ord_total[n]][AMR_TAG] == 1) && (block[n_ord_total[n]][AMR_LEVEL1] < max_levels - 1) && block[n_ord_total[n]][AMR_ACTIVE] == 1){ //If satisfy refinement criterion and smaller than maximum levels
				block[n_ord_total[n]][AMR_TAG] = 1;
				
				//Refine one level less near black hole
				level = block[n_ord_total[n]][AMR_LEVEL1];

				#if(SPHERICAL || SPHERICAL_GR)
					#if(NB_1<20 || !DEREFINE_POLE)
					if ((block[n_ord_total[n]][AMR_LEVEL1] == 0 && block[n_ord_total[n]][AMR_COORD1] < 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 1 && block[n_ord_total[n]][AMR_COORD1] < 2 + 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 2 && block[n_ord_total[n]][AMR_COORD1] < 6 + 1)
						|| (block[n_ord_total[n]][AMR_LEVEL1] == 3 && block[n_ord_total[n]][AMR_COORD1] < 14 + 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 4 && block[n_ord_total[n]][AMR_COORD1] < 30 + 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 5 && block[n_ord_total[n]][AMR_COORD1] < 62 + 1)){
						block[n_ord_total[n]][AMR_TAG] = 0;
					}
					#else
					if ( (block[n_ord_total[n]][AMR_LEVEL1] == 0 && block[n_ord_total[n]][AMR_COORD1] < 4) || (block[n_ord_total[n]][AMR_LEVEL1] == 1 && block[n_ord_total[n]][AMR_COORD1] < 10) || (block[n_ord_total[n]][AMR_LEVEL1] == 2 && block[n_ord_total[n]][AMR_COORD1] < 26)
						|| (block[n_ord_total[n]][AMR_LEVEL1] == 3 && block[n_ord_total[n]][AMR_COORD1] < 42 + 2) || (block[n_ord_total[n]][AMR_LEVEL1] == 4 && block[n_ord_total[n]][AMR_COORD1] < 96 + 2) || (block[n_ord_total[n]][AMR_LEVEL1] == 5 && block[n_ord_total[n]][AMR_COORD1] < 196 + 2)){
						block[n_ord_total[n]][AMR_TAG] = 0;
					}
					#endif
				#endif			

				#if(DEREFINE_POLE)
				//var = NB_2 / 3-1;
				//if ((block[n_ord_total[n]][AMR_LEVEL2] == 0 && block[n_ord_total[n]][AMR_COORD2] <= var) || (block[n_ord_total[n]][AMR_LEVEL2] == 1 && block[n_ord_total[n]][AMR_COORD2] <= 2 + var*pow(1 + REF_2, 1)) || (block[n_ord_total[n]][AMR_LEVEL2] == 2 && block[n_ord_total[n]][AMR_COORD2] <= 6 + var*pow(1 + REF_2, 2))
				//	|| (block[n_ord_total[n]][AMR_LEVEL2] == 3 && block[n_ord_total[n]][AMR_COORD2] == 14 + var*pow(1 + REF_2, 3)) || (block[n_ord_total[n]][AMR_LEVEL2] == 4 && block[n_ord_total[n]][AMR_COORD2] == 30 + var*pow(1 + REF_2, 4)) || (block[n_ord_total[n]][AMR_LEVEL2] == 5 && block[n_ord_total[n]][AMR_COORD2] == 62 + var*pow(1 + REF_2, 5))){
				////	if ((block[n_ord_total[n]][AMR_LEVEL1] == 0 && block[n_ord_total[n]][AMR_COORD1] <= 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 1 && block[n_ord_total[n]][AMR_COORD1] <= 4) || (block[n_ord_total[n]][AMR_LEVEL1] == 2 && block[n_ord_total[n]][AMR_COORD1] <= 10)
				//		|| (block[n_ord_total[n]][AMR_LEVEL1] == 3 && block[n_ord_total[n]][AMR_COORD1] == 22) || (block[n_ord_total[n]][AMR_LEVEL1] == 4 && block[n_ord_total[n]][AMR_COORD1] == 46) || (block[n_ord_total[n]][AMR_LEVEL1] == 5 && block[n_ord_total[n]][AMR_COORD1] == 94)) {
				//		block[n_ord_total[n]][AMR_TAG] = 0;
				//	}
				//}
				//if ((block[n_ord_total[n]][AMR_LEVEL2] == 0 && (NB_2-block[n_ord_total[n]][AMR_COORD2]) <= var) || (block[n_ord_total[n]][AMR_LEVEL2] == 1 && (NB_2*pow(1 + REF_2, 1) - block[n_ord_total[n]][AMR_COORD2]) <= 2 + var*pow(1 + REF_2, 1)) || (block[n_ord_total[n]][AMR_LEVEL2] == 2 && (NB_2*pow(1 + REF_2, 2) - block[n_ord_total[n]][AMR_COORD2]) <= 6 + var*pow(1 + REF_2, 2))
				//	|| (block[n_ord_total[n]][AMR_LEVEL2] == 3 && (NB_2*pow(1 + REF_2, 3) - block[n_ord_total[n]][AMR_COORD2]) == 14 + var*pow(1 + REF_2, 3)) || (block[n_ord_total[n]][AMR_LEVEL2] == 4 && (NB_2*pow(1 + REF_2, 4) - block[n_ord_total[n]][AMR_COORD2]) == 30 + var*pow(1 + REF_2, 4)) || (block[n_ord_total[n]][AMR_LEVEL2] == 5 && (NB_2*pow(1 + REF_2, 5) - block[n_ord_total[n]][AMR_COORD2]) == 62 + var*pow(1 + REF_2, 5))) {
				//	if ((block[n_ord_total[n]][AMR_LEVEL1] == 0 && block[n_ord_total[n]][AMR_COORD1] <= 1) || (block[n_ord_total[n]][AMR_LEVEL1] == 1 && block[n_ord_total[n]][AMR_COORD1] <= 4) || (block[n_ord_total[n]][AMR_LEVEL1] == 2 && block[n_ord_total[n]][AMR_COORD1] <= 10)
				//		|| (block[n_ord_total[n]][AMR_LEVEL1] == 3 && block[n_ord_total[n]][AMR_COORD1] == 22) || (block[n_ord_total[n]][AMR_LEVEL1] == 4 && block[n_ord_total[n]][AMR_COORD1] == 46) || (block[n_ord_total[n]][AMR_LEVEL1] == 5 && block[n_ord_total[n]][AMR_COORD1] == 94)) {
				//		block[n_ord_total[n]][AMR_TAG] = 0;
				//	}
				//}
				#endif

				if (block[n_ord_total[n]][AMR_TAG] >= 1){
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

		if (tag != 0 && n_active_total<max_blocks){
			MPI_Barrier(mpi_cartcomm);
			if(rank==0) fprintf(stderr, "Intermediate load balance! \n");
			balance_load();
			#if(GPU_ENABLED)
			balance_load_gpu();
			#endif
			pre_refine();
		}
	} while (tag != 0 && n_active_total<max_blocks && count<10);

	if (tag == 1){
		if(rank==0) fprintf(stderr, "Maximum number of blocks exceeded. Please select more nodes or adjust refinement criterion! \n");
		exit(0);
	}

	//First make sure all nodes have the same ref_val
	if (one_block_refined == 1) synch_refcrit();

	count = 0;
	gpu_counter = 0;
	int pole_tag = 0;
	do{
		count++;
		tag = 0;
		one_block_derefined = 0;

		for (n = 0; n < n_active_total; n++){
			//derefine
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[n_ord_total[n]][AMR_LEVEL1] > 0){
				block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = -1; //Tag for derefinement

				for (i = AMR_CHILD1; i <= AMR_CHILD8; i++){
					if (block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_REFINED] == 1)block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1; //If one of the children of the parent block is refined
					if (ref_val[block[block[n_ord_total[n]][AMR_PARENT]][i]] > 0.5*REFINEMENT_CUTOFF) block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 1; //Except if one of the children does satisfy the refinement criterion
				}
			}

			#if(SPHERICAL || SPHERICAL_GR)
			//Do not derefine other block around pole
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] > 0 && (block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == 0 || block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == NB_2*pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) - 1)){
				for (z = 0; z < NB_3*pow(1 + REF_3, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]); z++){
					block[AMR_coord_linear2(block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2]
						/ pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]), block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD1], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2], z)][AMR_TAG] = 2;
				}
			}
			#endif
		}
		
		do{
			tag2 = 0;
			for (n = 0; n < n_active_total; n++){
				//do not derefine if required for proper nesting
				for (i = AMR_NBR1; i <= AMR_CORN12; i++){
					if (block[n_ord_total[n]][i] == NB && block[n_ord_total[n]][AMR_PARENT]>=0){
						if (block[n_ord_total[n]][i + (AMR_NBR1P - AMR_NBR1)] >= 0 && block[block[n_ord_total[n]][i + (AMR_NBR1P - AMR_NBR1)]][AMR_TAG] >= 1){
							if (block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] != 2){
								tag2 = 1;
							}
							block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 2;
						}
					}
					else if(block[n_ord_total[n]][AMR_PARENT] >= 0) {
						if (block[n_ord_total[n]][i] >= 0 && block[block[n_ord_total[n]][i]][AMR_TAG] >= 1){
							if (block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] != 2){
								tag2 = 1;
							}
							block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 2;
						}
					}
				}

				#if(SPHERICAL || SPHERICAL_GR)
				//Do not derefine other block around pole
				if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] > 0 && (block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == 0 || block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2] == NB_2*pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) - 1)){
					for (z = 0; z < NB_3*pow(1 + REF_3, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]); z++){
						if (block[AMR_coord_linear2(block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2]
							/ pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]), block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD1], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2], z)][AMR_TAG] != 2){
							tag2 = 1;
						}
						block[AMR_coord_linear2(block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2]
							/ pow(1 + REF_2, block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]), block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD1], block[block[n_ord_total[n]][AMR_PARENT]][AMR_COORD2], z)][AMR_TAG] = 2;
					}
				}
				#endif
			}
		} while (tag2);

		//Detag if load balancing required as intermediate step
		//#pragma omp parallel for schedule(dynamic,1) private(n, node)
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]) {
				//#pragma omp critical
				//{
					node = block[n_ord_total[n]][AMR_NODE];
					if (NODE_global[node*N_GPU + block[n_ord_total[n]][AMR_GPU]] < max_blocks/numtasks + (1 + REF_1)*(1 + REF_2)*(1 + REF_1) - 1) {
						for (i1 = 0; i1 < 1 + REF_1; i1++)for (i2 = 0; i2 < 1 + REF_2; i2++)for (i3 = 0; i3 < 1 + REF_3; i3++) {
							i = AMR_CHILD1 + i1 * 4 + i2 * 2 + i3;
							if (((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL1] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL1]) % (1 + REF_1) >= i1) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL2] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) % (1 + REF_2) >= i2) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]) % (1 + REF_3) >= i3)) {
								n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
								if (block[n_send][AMR_NODE] != node) {
									NODE_global[node*N_GPU + block[n_ord_total[n]][AMR_GPU]]++;
								}
							}
						}
					}
					else {
						block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 0;
						tag = 1;
					}
				//}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Send/REceive blocks
		//#pragma omp parallel for schedule(dynamic,1) private(n, node)
		for (n = 0; n < n_active_total; n++) {
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]) {
					//Then derefine and set corresponding tag and timelevel
					if (one_block_derefined == 0) {
						prolong_grid();
						one_block_derefined = 1;
					}				
				//#pragma omp critical
				//{
					node = block[n_ord_total[n]][AMR_NODE];
					//Send block using non-blocking send
					for (i1 = 0; i1 < 1 + REF_1; i1++)for (i2 = 0; i2 < 1 + REF_2; i2++)for (i3 = 0; i3 < 1 + REF_3; i3++) {
						i = AMR_CHILD1 + i1 * 4 + i2 * 2 + i3;
						if (((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL1] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL1]) % (1 + REF_1) >= i1) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL2] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) % (1 + REF_2) >= i2) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]) % (1 + REF_3) >= i3)) {
							n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
							if (block[n_send][AMR_NODE] != node) {
								rc = 0;
								if (block[n_send][AMR_NODE] == rank) {
									rc += MPI_Isend(&p[nl[n_send]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (3 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][598]);
									#if STAGGERED
									rc += MPI_Isend(&ps[nl[n_send]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (4 * NB_LOCAL + block[n_send][AMR_NUMBER]) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[nl[n_send]][597]);
									#endif
								}
								if (rc != 0)fprintf(stderr, "Error in MPI in derefine \n");

								if (node == rank) {
									//Allocate memory for active blocks on node
									temp_p[n_send] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
									temp_ps[n_send] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
									if (block[n_send][AMR_NODE] >= 0) {
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
				//}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Clean up on sending side
		//#pragma omp parallel for schedule(dynamic,1) private(n, node)
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]) {
				#pragma omp critical
				//{
					node = block[n_ord_total[n]][AMR_NODE];
					for (i1 = 0; i1 < 1 + REF_1; i1++)for (i2 = 0; i2 < 1 + REF_2; i2++)for (i3 = 0; i3 < 1 + REF_3; i3++) {
						i = AMR_CHILD1 + i1 * 4 + i2 * 2 + i3;
						//Then use MPI_wait to clean up data that has been sent
						if (((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL1] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL1]) % (1 + REF_1) >= i1) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL2] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) % (1 + REF_2) >= i2) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]) % (1 + REF_3) >= i3)) {
							n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
							if (block[n_send][AMR_NODE] != node) {
								if (block[n_send][AMR_NODE] == rank) {
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
				//}
			}
		}

		//First make sure all blocks needed for derefinement are on the same node are on the same node: Allocate arrays on receiving side
		//#pragma omp parallel for schedule(dynamic,1) private(n)
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_PARENT] >= 0 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] == -1 && block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] == n_ord_total[n]) {
				//#pragma omp critical
				//{
					node = block[n_ord_total[n]][AMR_NODE];
					for (i1 = 0; i1 < 1 + REF_1; i1++)for (i2 = 0; i2 < 1 + REF_2; i2++)for (i3 = 0; i3 < 1 + REF_3; i3++) {
						i = AMR_CHILD1 + i1 * 4 + i2 * 2 + i3;
						//Then initialize sent data on receiving node
						if (((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL1] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL1]) % (1 + REF_1) >= i1) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL2] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL2]) % (1 + REF_2) >= i2) && ((block[block[block[n_ord_total[n]][AMR_PARENT]][i]][AMR_LEVEL3] - block[block[n_ord_total[n]][AMR_PARENT]][AMR_LEVEL3]) % (1 + REF_3) >= i3)) {
							n_send = block[block[n_ord_total[n]][AMR_PARENT]][i];
							if (block[n_send][AMR_NODE] != node) {
								if (node == rank) {
									if (block[n_send][AMR_NODE] >= 0) {
										MPI_Wait(&boundreqstemp1[n_send], &Statbound[0][10]);
										#if STAGGERED
										MPI_Wait(&boundreqstemp2[n_send], &Statbound[0][11]);
										#endif
									}
									set_arrays(n_send);
									set_grid(n_send);
									#pragma omp parallel for schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)  private(i, j, z, k)
									ZSLOOP3D(N1_GPU_offset[n_send] - N1G, N1_GPU_offset[n_send] + BS_1 - 1 + N1G, N2_GPU_offset[n_send] - N2G, N2_GPU_offset[n_send] + BS_2 - 1 + N2G, N3_GPU_offset[n_send] - N3G, N3_GPU_offset[n_send] + BS_3 - 1 + N3G) {
										PLOOP p[nl[n_send]][index_3D(n_send, i, j, z)][k] = temp_p[n_send][index_3D(n_send, i, j, z)][k];
										for (k = 0; k < NDIM; k++) ps[nl[n_send]][index_3D(n_send, i, j, z)][k] = temp_ps[n_send][index_3D(n_send, i, j, z)][k];
									}
									free(temp_p[n_send]);
									free(temp_ps[n_send]);
									#if(GPU_ENABLED || GPU_DEBUG )
									if (mem_spot_gpu[nl[n_send]] != -1) gpu_choice = mem_spot_gpu[nl[n_send]];
									else {
										gpu_choice = gpu_counter%N_GPU;
										gpu_counter++;
									}
									set_arrays_GPU(n_send, rank%numdevices);
									GPU_write(n_send);
									#endif
								}
							}
						}
						block[n_send][AMR_NODE] = node;
					}
					block[block[n_ord_total[n]][AMR_PARENT]][AMR_NODE] = node;
					derefine(block[n_ord_total[n]][AMR_PARENT]);
					pole_tag = 1;
					block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 0;
				//}
			}
		}

		//Fixate timelevel around pole and do post-refinement steps
		if (pole_tag == 1) timelevel_fixate();
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
	
	//Start very conservatively
	dt /= 2.;
	//for (n = 0; n < n_active_total; n++){
	//	block[n_ord_total[n]][AMR_TIMELEVEL] = 1;
	//}
	set_corners(0);

	//Set boundaries on GPU
	#if(GPU_ENABLED || GPU_DEBUG )
	for(n=0;n<n_active;n++) alloc_bounds_GPU(n_ord[n]);
	GPU_boundprim(1);
	GPU_boundprim(1);
	#endif

	end1 = time(NULL);
	if (rank == 0) fprintf(stderr, "Runtime load balance: %f \n", (double)(end1 - begin1));
}

//This function derefines in phi near the pole
int derefine_pole(void){
	int i, j, z, l, ni, nj, nz, u, n;
	//if (REF_3 != 1 || REF_1 == 1 || REF_2 == 1){
		//if(rank==0)fprintf(stderr, "Error! Derefinement near the pole works only for REF_1=0, REF_2=0, REF_3=1 \n");
		//exit(20);
		//return -1;
	//}
	if(rank==0)fprintf(stderr, "Derefining in phi by %d levels! \n", N_LEVELS_1D);

	if (NB_2 != 3 && NB_2 != 6 && NB_2 != 12 && NB_2 != 24 && NB_2 != 48 && NB_2 != 96){
		if (rank == 0)fprintf(stderr, "For derefinement near the pole chose NB_2 6, 12, 24, 48, 96 for 1, 2, 3, 4 levels of derefinement near the pole! \n");
		exit(20);
		return -1;
	}
	#if(GPU_ENABLED || GPU_DEBUG )
	for (n = 0; n<n_active; n++) free_bound_gpu(n_ord[n]);
	#endif
	//if (calc_mem(NB_1*NB_2*NB_3*pow(2., N_LEVELS_1D - 1)) > ((double)numtasks*(double)(numdevices)* 4. * (pow(10., 9.))) && rank == 1) fprintf(stderr, "You are exceeding the maximum memory size of 4 GB per GPU by refining too many blocks! Code will probably segfault, choose a bigger cluster \n");
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
	for (n = 0; n<n_active; n++) alloc_bounds_GPU(n_ord[n]);
	GPU_boundprim(1);
	#endif
	set_corners(0);
	return 1;
}

//Set row major order in case of no derfinement near pole
void rm_order2(void){
	int n, l, i, j, z, number;
	int counter = 0;
	int counter2 = 0;
	for (n = 0; n <= n_max; n++){
		AMR_coord_cart_RM(n, &l, &i, &j, &z); //transform to cartesian grid coordinates
		number = AMR_coord_linear2(l,j/(int)pow(1+REF_2,block[n][AMR_LEVEL2]), i, j, z); //transform to normal lineair ordering
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

	#if(CARTESIAN)
	if (block[n][AMR_COORD1] / pow(1 + REF_1, block[n][AMR_LEVEL1]) == 1 && block[n][AMR_COORD2] / pow(1 + REF_2, block[n][AMR_LEVEL2]) == 1 && block[n][AMR_COORD3] / pow(1 + REF_3, block[n][AMR_LEVEL3]) == 1) {
		ref_val = 1.01 * REFINEMENT_CUTOFF;
	}
	#elif(CARTESIAN_GR)
	if (NB_1 < 4 || NB_2 < 4 || NB_3 < 4) {
		if (rank == 0) fprintf(stderr, "Warning: Cartesian grid set up improperly");
		exit(0);
	}

	int min_i, min_j, min_z;
	int max_i, max_j, max_z;

	//Calculate minimum index
	min_i = NB_1 * pow(1 + REF_1, block[n][AMR_LEVEL]) / 2 - NB_1 * pow(1 + REF_1, 1 - 2) / 2;
	min_j = NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL]) / 2 - NB_2 * pow(1 + REF_2, 1 - 2) / 2;
	min_z = NB_3 * pow(1 + REF_3, block[n][AMR_LEVEL]) / 2 - NB_3 * pow(1 + REF_3, 1 - 2) / 2;

	//Calculate maximum index
	max_i = NB_1 * pow(1 + REF_1, block[n][AMR_LEVEL]) / 2 + NB_1 * pow(1 + REF_1, 1 - 2) / 2;
	max_j = NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL]) / 2 + NB_2 * pow(1 + REF_2, 1 - 2) / 2;
	max_z = NB_3 * pow(1 + REF_3, block[n][AMR_LEVEL]) / 2 + NB_3 * pow(1 + REF_3, 1 - 2) / 2;

	//Calculate extra cartesian levels
	if (block[n][AMR_LEVEL] <= N_LEVELS_CART - 1 &&  block[n][AMR_COORD1] >= min_i && block[n][AMR_COORD1] < max_i && block[n][AMR_COORD2] >= min_j && block[n][AMR_COORD2] < max_j && block[n][AMR_COORD3] >= min_z && block[n][AMR_COORD3] < max_z) {
	//if (block[n][AMR_LEVEL] < N_LEVELS_3D - 1 &&  block[n][AMR_COORD1] >= min_i && block[n][AMR_COORD1] < max_i && block[n][AMR_COORD2] >= min_j && block[n][AMR_COORD2] < max_j && block[n][AMR_COORD3] >= min_z && block[n][AMR_COORD3] < max_z) {
		ref_val = 1.1 * REFINEMENT_CUTOFF;
	}
	else {
		ref_val = 0.6 * REFINEMENT_CUTOFF;
	}
	#elif(REFINE_GIBWA)
	if (block[n][AMR_LEVEL2] == 0) {
		if (block[n][AMR_COORD1] >= 1 && block[n][AMR_COORD1] < 7) {
			if (block[n][AMR_COORD2] == 5 || block[n][AMR_COORD2] == 6)ref_val = 100.0;
		}
	}
	if (block[n][AMR_LEVEL2] == 1) {
		if (block[n][AMR_COORD1] >= 4 && block[n][AMR_COORD1] < 10) {
			if (block[n][AMR_COORD2] == 11 || block[n][AMR_COORD2] == 12) ref_val = 100.0;
		}
		else if (block[n][AMR_COORD1] >= 2 && block[n][AMR_COORD1] < 14) {
			if (block[n][AMR_COORD2] >= 10 && block[n][AMR_COORD2] < 14) ref_val = 0.6 * REFINEMENT_CUTOFF;
		}
	}
	if (block[n][AMR_LEVEL2] == 2) {
		if (block[n][AMR_COORD1] >= 8 && block[n][AMR_COORD1] < 20) {
			if (block[n][AMR_COORD2] >= 22 && block[n][AMR_COORD2] < 26) ref_val = 0.6 * REFINEMENT_CUTOFF;
		}
	}
	#elif(REFINE_JET)
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);

			if ((r > 50.0) && (phi>0.5*3.14) && (phi<1.5*3.14)){
				get_geometry(n, i, j, z, CENT, &geom);
				get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);
				bsq = bsq_calc(p[nl[n]][index_3D(n, i, j, z)], &geom);

				//Ore's criterion
				//if ((p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.1) || (bsq/p[nl[n]][index_3D(n, i, j, z)][RHO] > 1.0)) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
				//else if ((p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.05) || (bsq / p[nl[n]][index_3D(n, i, j, z)][RHO] > 0.5)) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				//if ((ref_val > REFINEMENT_CUTOFF) && (block[n][AMR_LEVEL1] == 1) && (r < 250.)) ref_val = 0.51 * REFINEMENT_CUTOFF;
				//if ((ref_val > REFINEMENT_CUTOFF) && (block[n][AMR_LEVEL1] == 2) && (r < 1000.)) ref_val = 0.51 * REFINEMENT_CUTOFF;

				//Matthew's criterion
				if (log10(q.ucon[0]) > 0.5 || log10(bsq / p[nl[n]][index_3D(n, i, j, z)][RHO]) > 0.3 || log10(p[nl[n]][index_3D(n, i, j, z)][UU] / p[nl[n]][index_3D(n, i, j, z)][RHO]) > -0.2) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
				else if (log10(q.ucon[0]) > 0.25 || log10(bsq / p[nl[n]][index_3D(n, i, j, z)][RHO]) > 0.15 || log10(p[nl[n]][index_3D(n, i, j, z)][UU] / p[nl[n]][index_3D(n, i, j, z)][RHO]) > -0.4) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				//if (log10(bsq / p[nl[n]][index_3D(n, i, j, z)][RHO]) > 0.3) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
				//else if (log10(bsq / p[nl[n]][index_3D(n, i, j, z)][RHO]) > 0.15) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				if ((ref_val > REFINEMENT_CUTOFF) && (block[n][AMR_LEVEL1] == 1) && (r < 250.)) ref_val = 0.51 * REFINEMENT_CUTOFF;
				if ((ref_val > REFINEMENT_CUTOFF) && (block[n][AMR_LEVEL1] == 2) && (r < 1000.)) ref_val = 0.51 * REFINEMENT_CUTOFF;
				if ((ref_val > REFINEMENT_CUTOFF) && (n_active_total > 0.95 * max_blocks/numtasks)) ref_val = 0.51 * REFINEMENT_CUTOFF;
			}
		}
	}
	#elif(REFINE_GRB) // jet, inner cocoon and disk maintain high entropy
	int index;
	float block_angle;
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if(block[n][AMR_LEVEL1] >= BASE_LEVELS){
				if (r > 3.1) {
					//Calc misc quantities
					get_geometry(n, i, j, z, CENT, &geom);
					get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);
					bsq = bsq_calc(p[nl[n]][index_3D(n, i, j, z)], &geom);

					//Matthew's new refinement criterion
					//Convert indices to appropriate format
					index = i / pow(1 + REF_1, block[n][AMR_LEVEL1]);

					//Calculate opening angle of a single block
					block_angle = M_PI / (NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL2]));

					//Check if cell is part of the jet; If so set the cell up for refinement
					#if (DOHELM && DOHELM_TEMPERATURE)
					double pres, u;
					eos_mode_rhotemp_pres_u(p[nl[n]][index_3D(n, i, j, z)][RHO], p[nl[n]][index_3D(n, i, j, z)][UU], p[nl[n]][index_3D(n, i, j, z)][YE], &pres, &u, 
						#if (DONUCLEAR)
						, p[nl[n]][index_3D(n, i, j, z)][XALPHA], p[nl[n]][index_3D(n, i, j, z)][XATM]
						#endif
						);
					if (u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > HIGH_CUTOFF && t>t_ff)
					#else
					if (p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > HIGH_CUTOFF && t>t_ff)
					#endif
					{
						if (TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) {
								if ((jet_angle1[index] / block_angle) < BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((jet_angle1[index] / block_angle) >= BLOCKS_PER_THETA_JET && (jet_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((jet_angle2[index] / block_angle) < BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((jet_angle2[index] / block_angle) >= BLOCKS_PER_THETA_JET && (jet_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
						else {
							if (phi > 0.5 * M_PI && phi < 1.5 * M_PI) {
								if ((jet_angle1[index] / block_angle) < BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((jet_angle1[index] / block_angle) > BLOCKS_PER_THETA_JET && (jet_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((jet_angle2[index] / block_angle) < BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((jet_angle2[index] / block_angle) > BLOCKS_PER_THETA_JET && (jet_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}		
					}

					#if (DOHELM && DOHELM_TEMPERATURE)
					if ((u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) < HIGH_CUTOFF && u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.5*HIGH_CUTOFF) && t>t_ff) {
					#else
					if ((p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) < HIGH_CUTOFF && p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.5*HIGH_CUTOFF) && t>t_ff) {
					#endif
						if (TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) {
								if ((jet_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((jet_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
						else {
							if (phi > 0.5 * M_PI && phi < 1.5 * M_PI) {
								if ((jet_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((jet_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_JET) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
					}
				
					//Check if cell is part of the coccoon; If so set the cell up for refinement
					#if (DOHELM && DOHELM_TEMPERATURE)
					if (u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > LOW_CUTOFF && t>t_ff)
					#else 
					if (p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > LOW_CUTOFF && t>t_ff)
					#endif
					{
						if (TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) {
								if ((coccoon_angle1[index] / block_angle) < BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((coccoon_angle1[index] / block_angle) >= BLOCKS_PER_THETA_COCCOON && (coccoon_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((coccoon_angle2[index] / block_angle) < BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((coccoon_angle2[index] / block_angle) >= BLOCKS_PER_THETA_COCCOON && (coccoon_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
						else {
							if (phi > 0.5 * M_PI && phi < 1.5 * M_PI) {
								if ((coccoon_angle1[index] / block_angle) < BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((coccoon_angle1[index] / block_angle) > BLOCKS_PER_THETA_COCCOON && (coccoon_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((coccoon_angle2[index] / block_angle) < BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
								else if ((coccoon_angle2[index] / block_angle) > BLOCKS_PER_THETA_COCCOON && (coccoon_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
					}
					#if (DOHELM && DOHELM_TEMPERATURE)
					if ((u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) < LOW_CUTOFF && u / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.5*LOW_CUTOFF) && t>t_ff) {
					#else
					if ((p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) < LOW_CUTOFF && p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA) > 0.5*LOW_CUTOFF) && t>t_ff) {
					#endif
						if (TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) {
								if ((coccoon_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((coccoon_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
						else {
							if (phi > 0.5 * M_PI && phi < 1.5 * M_PI) {
								if ((coccoon_angle1[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
							else {
								if ((coccoon_angle2[index] / block_angle) < 3 * BLOCKS_PER_THETA_COCCOON) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
							}
						}
					}
					//Old refinement criterion
					//if (p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], 1.33) > 1 && p[nl[n]][index_3D(n, i, j, z)][UU] * r * r > 1e-5) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
					//if (p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], 1.33) < 1 && p[nl[n]][index_3D(n, i, j, z)][UU] / pow(p[nl[n]][index_3D(n, i, j, z)][RHO], 1.33) > 0.5) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
					//if ((ref_val > REFINEMENT_CUTOFF) && ((block[n][AMR_LEVEL1] - BASE_LEVELS) == 0) && (r < R0)) ref_val = 0.51 * REFINEMENT_CUTOFF; //Disable first refinement level below r=R0
					//if ((ref_val > REFINEMENT_CUTOFF) && ((block[n][AMR_LEVEL1] - BASE_LEVELS) == 1) && (r < R1)) ref_val = 0.51 * REFINEMENT_CUTOFF;
					//if ((ref_val > REFINEMENT_CUTOFF) && ((block[n][AMR_LEVEL1] - BASE_LEVELS) == 2) && (r < R2)) ref_val = 0.51 * REFINEMENT_CUTOFF;
					//if ((ref_val > REFINEMENT_CUTOFF) && ((block[n][AMR_LEVEL1] - BASE_LEVELS) == 3) && (r < R3)) ref_val = 0.51 * REFINEMENT_CUTOFF;
					//if ((ref_val > REFINEMENT_CUTOFF) && ((block[n][AMR_LEVEL1] - BASE_LEVELS) == 4) && (r < R4)) ref_val = 0.51 * REFINEMENT_CUTOFF; //Disable fifth refinement level below r=R4

					//Do not derefine base grid
					if ((block[n][AMR_LEVEL1] == BASE_LEVELS)) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				}
				else{
					ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				}
			}
			else {
				if (r > 3.1) {
					ref_val = 1.01 * REFINEMENT_CUTOFF;
				}
			}
			
			//No refinement near black hole; even in case of derefine_pole
			/*if(ref_val>=REFINEMENT_CUTOFF){
				#if(DEREFINE_POLE==0 || BS_1<32)
				if ((block[n][AMR_LEVEL1] == 0 && block[n][AMR_COORD1] < 2) || (block[n][AMR_LEVEL1] == 1 && block[n][AMR_COORD1] < 4 + 2) || (block[n][AMR_LEVEL1] == 2 && block[n][AMR_COORD1] < 12 + 2)
					|| (block[n][AMR_LEVEL1] == 3 && block[n][AMR_COORD1] < 28 + 2) || (block[n][AMR_LEVEL1] == 4 && block[n][AMR_COORD1] < 60 + 2) || (block[n][AMR_LEVEL1] == 5 && block[n][AMR_COORD1] < 124 + 2)
					|| (block[n][AMR_LEVEL1] == 6 && block[n][AMR_COORD1] < 252 + 2) || (block[n][AMR_LEVEL1] == 7 && block[n][AMR_COORD1] < 508 + 2) || (block[n][AMR_LEVEL1] == 8 && block[n][AMR_COORD1] < 1020 + 2)){
					ref_val = 0.51 * REFINEMENT_CUTOFF;
				}
				#else //Required to prevent jumps in AMR near pole due to 1D refinement; Another possibility is to prohibit the code from refining blocks neighboring the block touching the pole
				if ( (block[n][AMR_LEVEL1] == 0 && block[n][AMR_COORD1] < 4) || (block[n][AMR_LEVEL1] == 1 && block[n][AMR_COORD1] < 10) || (block[n][AMR_LEVEL1] == 2 && block[n][AMR_COORD1] < 26)
					|| (block[n][AMR_LEVEL1] == 3 && block[n][AMR_COORD1] < 42 + 2) || (block[n][AMR_LEVEL1] == 4 && block[n][AMR_COORD1] < 96 + 2) || (block[n][AMR_LEVEL1] == 5 && block[n][AMR_COORD1] < 196 + 2)){
					ref_val = 0.51 * REFINEMENT_CUTOFF;
				}
				#endif
			}*/
		}
	}
	#elif(REFINE_THIN)
	//Loop over grid and determine cell-by-cell if to refine
	if (block[n][AMR_NODE] == rank) {
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			get_geometry(n, i, j, z, CENT, &geom);
			get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);
			bsq = bsq_calc(p[nl[n]][index_3D(n, i, j, z)], &geom);
			double rho = p[nl[n]][index_3D(n, i, j, z)][RHO];
			#if(RAD_M1)
			double ptot = ((GAMMA - 1.) * p[nl[n]][index_3D(n, i, j, z)][UU]) + (1./3.*p[nl[n]][index_3D(n, i, j, z)][UU_RAD]) + 0.5 * bsq;
			double cs = sqrt(2.0 / M_PI * ptot / (GAMMA * p[nl[n]][index_3D(n, i, j, z)][UU] + (4. / 3. * p[nl[n]][index_3D(n, i, j, z)][UU_RAD]) + bsq + rho));
			#else
			double ptot = ((GAMMA - 1.) * p[nl[n]][index_3D(n, i, j, z)][UU]) + 0.5 * bsq;
			double cs = sqrt(2.0 / M_PI * ptot / (GAMMA * p[nl[n]][index_3D(n, i, j, z)][UU] + bsq + rho));
			#endif
			double v_kepler = r / (pow(r, 1.5) + a);
			double scaleheight = cs / v_kepler;
			double cells_per_scaleheight = scaleheight / M_PI * NB_2 * BS_2 * pow(1.0 + REF_2, block[n][AMR_LEVEL2]);

			//Real refinement criterion
			if ((cells_per_scaleheight < CELLS_PER_SCALEHEIGHT) && (rho>0.25*density_midplane[(int)(i/pow(1+REF_1,block[n][AMR_LEVEL1]))]) && (rho > 0.1) && (bsq / rho < 1.0) && (r < 400.0)) ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
			else if((cells_per_scaleheight >= CELLS_PER_SCALEHEIGHT) && (cells_per_scaleheight < 3.0*CELLS_PER_SCALEHEIGHT) && (rho > 0.05 * density_midplane[(int)(i / pow(1 + REF_1, block[n][AMR_LEVEL1]))]) && (rho>0.05) && (bsq / rho < 2.0) && (r<400.0)) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
		}
	}
	#elif(WHICHPROBLEM==DISRUPTION_PROBLEM)
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			enth=1.0+p[nl[n]][index_3D(n, i, j, z)][UU]*gam/p[nl[n]][index_3D(n, i, j, z)][RHO];
			if (p[nl[n]][index_3D(n, i, j, z)][RHO]*fabs(enth) > ref_val) ref_val = p[nl[n]][index_3D(n, i, j, z)][RHO]*enth;
		}
	}
	#elif(WHICHPROBLEM==THIN_PROBLEM)
	if (block[n][AMR_NODE] == rank) {
		//#pragma omp parallel for schedule(dynamic,1) private(i,j,z,X,r,th,phi)
		ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			//#pragma omp critical
			//{
			if ((p[nl[n]][index_3D(n, i, j, z)][RHO] * (r * (r > 25.) + r * (pow(25. / r, 3.0)) * (r <= 25.))) > ref_val && r < 150.) ref_val = (p[nl[n]][index_3D(n, i, j, z)][RHO] * (r * (r > 25.) + r * (pow(25. / r, 3.0)) * (r <= 25.)));
			//}
		}
	}
	#elif(WHICHPROBLEM==POSTMERGER_PROBLEM || WHICHPROBLEM==TORUS_PROBLEM || WHICHPROBLEM==COLLAPSAR)
	int index;
	float block_angle;
	if (block[n][AMR_NODE] == rank) {
	ZSLOOP3D(N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		if (block[n][AMR_LEVEL1] >= BASE_LEVELS) {
			//if (r > 17) {
			if (r > 4.) {
				//Calc misc quantities
				get_geometry(n, i, j, z, CENT, &geom);
				get_state(p[nl[n]][index_3D(n, i, j, z)], &geom, &q);
				bsq = bsq_calc(p[nl[n]][index_3D(n, i, j, z)], &geom);

				//Matthew's new refinement criterion
				//Convert indices to appropriate format
				//index = i / pow(1 + REF_1, block[n][AMR_LEVEL1]);

				//Calculate opening angle of a single block
				//block_angle = M_PI / (NB_2 * pow(1 + REF_2, block[n][AMR_LEVEL2]));

				//Check if cell is part of the jet; If so set the cell up for refinement
				double rho = p[nl[n]][index_3D(n, i, j, z)][RHO];
				if (rho > REFINEMENT_CUTOFF) {
					ref_val = MY_MAX(ref_val, 1.01 * REFINEMENT_CUTOFF);
				}
				else if (rho > 0.51 * REFINEMENT_CUTOFF) {
					ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
				}
				//Do not derefine base grid
				if (block[n][AMR_LEVEL1] == BASE_LEVELS) ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
			}
			else {
				ref_val = MY_MAX(ref_val, 0.51 * REFINEMENT_CUTOFF);
			}
		}
		else {
			//if (r > 17) {
			if (r > 4.) {
				ref_val = 1.01 * REFINEMENT_CUTOFF;
			}
		}

		//No refinement near black hole; even in case of derefine_pole
		/*if(ref_val>=REFINEMENT_CUTOFF){
			#if(DEREFINE_POLE==0 || BS_1<32)
			if ((block[n][AMR_LEVEL1] == 0 && block[n][AMR_COORD1] < 2) || (block[n][AMR_LEVEL1] == 1 && block[n][AMR_COORD1] < 4 + 2) || (block[n][AMR_LEVEL1] == 2 && block[n][AMR_COORD1] < 12 + 2)
				|| (block[n][AMR_LEVEL1] == 3 && block[n][AMR_COORD1] < 28 + 2) || (block[n][AMR_LEVEL1] == 4 && block[n][AMR_COORD1] < 60 + 2) || (block[n][AMR_LEVEL1] == 5 && block[n][AMR_COORD1] < 124 + 2)
				|| (block[n][AMR_LEVEL1] == 6 && block[n][AMR_COORD1] < 252 + 2) || (block[n][AMR_LEVEL1] == 7 && block[n][AMR_COORD1] < 508 + 2) || (block[n][AMR_LEVEL1] == 8 && block[n][AMR_COORD1] < 1020 + 2)){
				ref_val = 0.51 * REFINEMENT_CUTOFF;
			}
			#else //Required to prevent jumps in AMR near pole due to 1D refinement; Another possibility is to prohibit the code from refining blocks neighboring the block touching the pole
			if ( (block[n][AMR_LEVEL1] == 0 && block[n][AMR_COORD1] < 4) || (block[n][AMR_LEVEL1] == 1 && block[n][AMR_COORD1] < 10) || (block[n][AMR_LEVEL1] == 2 && block[n][AMR_COORD1] < 26)
				|| (block[n][AMR_LEVEL1] == 3 && block[n][AMR_COORD1] < 42 + 2) || (block[n][AMR_LEVEL1] == 4 && block[n][AMR_COORD1] < 96 + 2) || (block[n][AMR_LEVEL1] == 5 && block[n][AMR_COORD1] < 196 + 2)){
				ref_val = 0.51 * REFINEMENT_CUTOFF;
			}
			#endif
		}*/
		}
	}
	#else
	ref_val = 0.0;
	#endif
	return ref_val;
}

//Send refinement criterion across cluster
void synch_refcrit(void){
	int n, task, i, j, stride=1000, n_stride;
	int counter = 0, counter_old=0;
	double ref_val_local[NB];

	#if(REFINE_GRB)
	//Calculate opening angle jet if REFINE_GRB is set
	calc_opening_jet();
	calc_opening_coccoon();
	#endif

	#if(REFINE_THIN)
	//Calculate midplane density
	calc_density_midplane();
	#endif

	//Calculate refinement criterion
	for (n = 0; n < n_active; n++) {
		ref_val[n_ord[n]] = calc_refcrit(n_ord[n]);
	}

	//Number of strides in messages of stride 1000
	n_stride = numtasks / stride + 1;

	for (j = 0; j < n_stride; j++) {
		//Transmit refinement criterion to all MPI ranks
		for (i = j * stride; i < MY_MIN(j * stride + stride, numtasks); i++) {
			if (rank == i) {
				for (n = 0; n < n_active_node[i]; n++) {
					ref_val_local[counter + n] = ref_val[n_ord_node[i][n]];
				}
			}
			rc = MPI_Ibcast(&ref_val_local[counter], n_active_node[i], MPI_DOUBLE, i, mpi_cartcomm, &request_timelevel[i]);
			counter += n_active_node[i];
		}

		counter = counter_old;
		for (i = j * stride; i < MY_MIN(j * stride + stride, numtasks); i++) {
			MPI_Wait(&request_timelevel[i], &Statbound[0][0]);

			if (rank != i) {
				for (n = 0; n < n_active_node[i]; n++) {
					ref_val[n_ord_node[i][n]] = ref_val_local[counter + n];
				}
			}
			counter += n_active_node[i];
		}
		counter_old = counter;
	}
}

void calc_density_midplane(void) {
	int i, j, n, z, index;
	double rho_squared[NB_1*BS_1], rho[NB_1*BS_1];
	struct of_geom geom;

	//Initialize midplane density to 0
	for (index = 0; index < NB_1 * BS_1; index++) {
		density_midplane[index] = 0.;
		rho_squared[index] = 0.;
		rho[index] = 0.;
	}

	//Calculate rho_squared and rho sum over grid
	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total[n]][AMR_NODE] == rank) {
			ZSLOOP3D(N1_GPU_offset[n_ord_total[n]], BS_1 + N1_GPU_offset[n_ord_total[n]] - 1, N2_GPU_offset[n_ord_total[n]], N2_GPU_offset[n_ord_total[n]] + BS_2 - 1, N3_GPU_offset[n_ord_total[n]], N3_GPU_offset[n_ord_total[n]] + BS_3 - 1) {
				//Convert indices to appropriate format
				index = i / pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]);

				//Calculate sum in grid of density and density squared
				if (index * pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]) == i) {
					get_geometry(n_ord_total[n], i, j, z, CENT, &geom);
					rho_squared[index] += geom.g * pow(p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][RHO], 2.0) * dx[nl[n_ord_total[n]]][1] * dx[nl[n_ord_total[n]]][2] *dx[nl[n_ord_total[n]]][3];
					rho[index]+= geom.g * p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][RHO] * dx[nl[n_ord_total[n]]][1] * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3];
				}
			}
		}
	}

	//Sum over MPI processes
	MPI_Allreduce(MPI_IN_PLACE, &(rho_squared[0]), NB_1 * BS_1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &(rho[0]), NB_1 * BS_1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);

	//Calculate midplane density
	for (index = 0; index < NB_1 * BS_1; index++) {
		density_midplane[index] = rho_squared[index] / rho[index];
	}
}

void calc_opening_jet(void) {
	//Calculate jet opening angle if refining on jet
	int i, j, n, z, task, index, index0;
	double X[NDIM], r, th, phi;
	struct of_geom geom;

	//Initialize jet angle to 0
	for (index = 0; index < NB_1 * BS_1; index++) {
		jet_angle1[index] = 0.;
		jet_angle2[index] = 0.;
	}

	//Loop over cells in block and add result to float jet_angle1 and jet_angle2 (upper and lower jets)
	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total[n]][AMR_NODE] == rank) {
			ZSLOOP3D(N1_GPU_offset[n_ord_total[n]], BS_1 + N1_GPU_offset[n_ord_total[n]] - 1, N2_GPU_offset[n_ord_total[n]], N2_GPU_offset[n_ord_total[n]] + BS_2 - 1, N3_GPU_offset[n_ord_total[n]], N3_GPU_offset[n_ord_total[n]] + BS_3 - 1) {
				//Convert indices to appropriate format
				index = i / pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]);
			
				//Check if index constraints are satisfied; e.g. block can be not relevant for refinement
				if (index * pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]) == i) {
					//Check if refinement criterion is satisfied
					coord(n_ord_total[n], i, j, z, CENT, X);
					bl_coord(X, &r, &th, &phi);

					if (p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][UU] / pow(p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][RHO], GAMMA) > HIGH_CUTOFF) {
						get_geometry(n_ord_total[n], i, j, z, CENT, &geom);
						if(TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) jet_angle1[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
							else jet_angle2[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
						}
						else {
							if (phi > 0.5 * M_PI && phi<1.5*M_PI) jet_angle1[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
							else jet_angle2[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
						}
					}
				}
			}
		}
	}

	//Calculate half opening angle
	//A=pi*(r*theta)^2
	//theta=sqrt(A/(pi*r^2))
	MPI_Allreduce(MPI_IN_PLACE, &(jet_angle1[0]), NB_1 * BS_1, MPI_FLOAT, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &(jet_angle2[0]), NB_1 * BS_1, MPI_FLOAT, MPI_SUM, mpi_cartcomm);

	for (index = 0; index < NB_1 * BS_1; index++) {
		jet_angle1[index] = sqrt(jet_angle1[index] / M_PI + 1.e-30);
		jet_angle2[index] = sqrt(jet_angle2[index] / M_PI + 1.e-30);
	}
	
	//Find jet head location
	float jet_angle1_max = 0.0, jet_angle2_max = 0.0;
	float  dx = (log(Rout) - startx[1]) / (NB_1 * BS_1);
	int index_max1, index_max2;
	for (index = 0; index < NB_1 * BS_1; index++) {
		//Calculate radius from internal coordinates assume x1=log(r)
		r = exp(startx[1] + index * dx);

		//Find maximum cylindrical radius
		if (jet_angle1[index] * r > jet_angle1_max) {
			jet_angle1_max = jet_angle1[index] * r;
			index_max1 = index;
		}
		if (jet_angle2[index] * r > jet_angle2_max) {
			jet_angle2_max = jet_angle2[index] * r;
			index_max2 = index;
		}
	}

	//Assume opening angle is constant after maximum in cylindrical radius
	for (index = index_max1; index < NB_1 * BS_1; index++) {
		index0 = MY_MAX(index - 1, 0);
		jet_angle1[index] = jet_angle1[index0];
	}
	for (index = index_max2; index < NB_1 * BS_1; index++) {
		index0 = MY_MAX(index - 1, 0);
		jet_angle2[index] = jet_angle2[index0];
	}
}

void calc_opening_coccoon(void) {
	//Calculate jet opening angle if refining on jet
	int i, j, n, z, task, index, index0;
	double X[NDIM], r, th, phi;
	struct of_geom geom;

	//Initialize jet angle to 0
	for (index = 0; index < NB_1 * BS_1; index++) {
		coccoon_angle1[index] = 0.;
		coccoon_angle2[index] = 0.;
	}

	//Loop over cells in block and add result to float coccoon_angle1 and coccoon_angle2 (upper and lower jets)
	for (n = 0; n < n_active_total; n++) {
		if (block[n_ord_total[n]][AMR_NODE] == rank) {
			ZSLOOP3D(N1_GPU_offset[n_ord_total[n]], BS_1 + N1_GPU_offset[n_ord_total[n]] - 1, N2_GPU_offset[n_ord_total[n]], N2_GPU_offset[n_ord_total[n]] + BS_2 - 1, N3_GPU_offset[n_ord_total[n]], N3_GPU_offset[n_ord_total[n]] + BS_3 - 1) {
				//Convert indices to appropriate format
				index = i / pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]);

				//Check if index constraints are satisfied; e.g. block can be not relevant for refinement
				if (index * pow(1 + REF_1, block[n_ord_total[n]][AMR_LEVEL1]) == i) {
					//Check if refinement criterion is satisfied
					coord(n_ord_total[n], i, j, z, CENT, X);
					bl_coord(X, &r, &th, &phi);

					if (p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][UU] / pow(p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][RHO], GAMMA) > LOW_CUTOFF && p[nl[n_ord_total[n]]][index_3D(n_ord_total[n], i, j, z)][UU] * r * r > 1.0e-5) {
						get_geometry(n_ord_total[n], i, j, z, CENT, &geom);
						if (TILT_ANGLE < 60) {
							if (th < M_PI / 2.0) coccoon_angle1[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
							else coccoon_angle2[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
						}
						else {
							if (phi > 0.5 * M_PI && phi < 1.5 * M_PI) coccoon_angle1[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
							else coccoon_angle2[index] += sqrt(geom.gcov[2][2]) * sqrt(geom.gcov[3][3]) * dx[nl[n_ord_total[n]]][2] * dx[nl[n_ord_total[n]]][3] / (r * r);
						}
					}
				}
			}
		}
	}

	//Calculate half opening angle
	//A=pi*(r*theta)^2
	//theta=sqrt(A/(pi*r^2))
	for (index = 0; index < NB_1 * BS_1; index++) {
		MPI_Allreduce(MPI_IN_PLACE, &(coccoon_angle1[index]), 1, MPI_FLOAT, MPI_SUM, mpi_cartcomm);
		coccoon_angle1[index] = sqrt(coccoon_angle1[index] / M_PI + 1.e-30);

		MPI_Allreduce(MPI_IN_PLACE, &(coccoon_angle2[index]), 1, MPI_FLOAT, MPI_SUM, mpi_cartcomm);
		coccoon_angle2[index] = sqrt(coccoon_angle2[index] / M_PI + 1.e-30);
	}

	//Find jet head location
	float coccoon_angle1_max = 0.0, coccoon_angle2_max = 0.0;
	float  dx = (log(Rout) - startx[1]) / (NB_1 * BS_1);
	int index_max1, index_max2;
	for (index = 0; index < NB_1 * BS_1; index++) {
		//Calculate radius from internal coordinates assume x1=log(r)
		r = exp(startx[1] + index * dx);

		//Find maximum cylindrical radius
		if (coccoon_angle1[index] * r > coccoon_angle1_max) {
			coccoon_angle1_max = coccoon_angle1[index] * r;
			index_max1 = index;
		}
		if (coccoon_angle2[index] * r > coccoon_angle2_max) {
			coccoon_angle2_max = coccoon_angle2[index] * r;
			index_max2 = index;
		}
	}

	//Assume opening angle is constant after maximum in cylindrical radius
	for (index = index_max1; index < NB_1 * BS_1; index++) {
		index0 = MY_MAX(index - 1, 0);
		coccoon_angle1[index] = coccoon_angle1[index0];
	}
	for (index = index_max2; index < NB_1 * BS_1; index++) {
		index0 = MY_MAX(index - 1, 0);
		coccoon_angle2[index] = coccoon_angle2[index0];
	}
}

//Calculates RAM requirements in bytes (conservatively)
double calc_mem(int n_blocks){
	return n_blocks*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * 1000;
}