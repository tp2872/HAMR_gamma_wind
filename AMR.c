#include "decs_MPI.h"

int AMR_coord_linear_RM(int level, int i, int j, int z);
void AMR_coord_cart_RM(int n, int *level, int *i, int *j, int *z);
int rm_order2(void);

void test_AMR(void){
	int n;
	//int level = 2;
	//int i = 1;
	//int j = 2;
	//int z = 2;
	//n=AMR_coord_linear(level, i, j, z);
	//printf("n:%d, ", n);
	//AMR_coord_cart(n, &level, &i, &j, &z);
	//printf("level:%d, i:%d, j:%d, z:%d \n", level,i,j,z);

	//set_AMR();
	for (n = 0; n < NB; n++){
		if (block[n][AMR_ACTIVE] == 1){
			printf("Block: %d, Level: %d NBR1: %d, NBR2: %d, NBR3: %d, NBR4: %d NBR5: %d, NBR6: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4], block[n][AMR_NBR5], block[n][AMR_NBR6]);
			printf("Block: %d, Level: %d coord1: %d, coord2: %d, coord3: %d \n", n, block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2], block[n][AMR_COORD3]);
		}
	}
	//printf("Block: %d, Level: %d NBR1: %d, NBR2: %d, NBR3: %d, NBR4: %d \n", n);

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


//returns the number of the communicator for avereging of E field across polar axis
int AMR_comm_linear(int level, int i, int j){
	int index[N_LEVELS], coord[NDIM], factor[N_LEVELS], u, y, n;

	if (i < 0 || j < 0){
		return -1;
	}

	for (y = 0; y < N_LEVELS; y++){
		factor[y] = 1;
		for (u = 0; u < N_LEVELS - y - 1; u++){
			factor[y] = factor[y] * pow(2, REF_1 + REF_2) + 1;
		}
	}

	index[0] = ((i - i % (int)pow(1 + REF_1, level)) / pow(1 + REF_1, level) * NB_2 + (j - j % (int)pow(1 + REF_2, level)) / pow(1 + REF_2, level));
		n = index[0] * factor[0];

	for (u = level; u > 0; u--){
		coord[1] = (i % (int)(pow(1 + REF_1, level - u + 1)) - (i % (int)pow(1 + REF_1, level - u))) / pow(2, level - u);
		coord[2] = (j % (int)(pow(1 + REF_2, level - u + 1)) - (j % (int)pow(1 + REF_2, level - u))) / pow(2, level - u);
		index[u] = coord[1] * (1 + REF_2) + coord[2]; //index of subblock within block in range [1,8] for refinement in 3 dimensions
		n += index[u] * factor[u] + 1;
	}
	return n;
}
	
void AMR_setgroup(void){
	int n, y;
	int y_max = 0;
	y_max = AMR_comm_linear(N_LEVELS - 1, pow(REF_1 + 1, N_LEVELS - 1)*NB_1 - 1, pow(REF_2 + 1, N_LEVELS - 1)*NB_2 - 1);

	for (n = 0; n <= n_max; n++){
		block[n][AMR_GROUP] = AMR_comm_linear(block[n][AMR_LEVEL], block[n][AMR_COORD1], block[n][AMR_COORD2]);
	}
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
	//printf("Factor: %d \n", factor[0]);
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
	//printf("Factor: %d \n", factor[0]);
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
		N1_GPU[n] = BS_1;
		N2_GPU_offset[n] = block[n][AMR_COORD2] * BS_2;
		N2_GPU[n] = BS_2;
		N3_GPU_offset[n] = block[n][AMR_COORD3] * BS_3;
		N3_GPU[n] = BS_3;
	}

	//Set the groups for which to sum the E1 values at the pole for transmissive boundary conditions
	AMR_setgroup();

	//Allocate array for image output
	//set_arrays_image();

	for (n = 0; n <= n_max; n++){
		set_points(n);

		//For the moment don't refine any block
		block[n][AMR_REFINED] = 0;

		//No node assigned yet
		block[n][AMR_NODE] = -1;

		//No special GPU assigned yet
		block[n][AMR_GPU] = local_rank%N_GPU;

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
	activate_blocks();
	set_corners();

	MPI_Barrier(mpi_cartcomm);

	balance_load();
}

#define MAX_BLOCKS (24*(184*22*34)/((BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)))
void balance_load(void){
	int i, node, tt, fp, ip, y, rem;
	int i1, j1, z1, k, n;
	int n_active_total_steps = 0;
	int steps_total_RM[NB];
	int NODE[NB];
	int timelevel_cutoff = AMR_MAXTIMELEVEL;
	#if(DEREFINE_POLE)
	rm_order();
	#else
	rm_order2();
	#endif

	int n_active_local_max = 0;
	int n_active_local_min = 1;

	do{
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
			NODE[i] = (steps_total_RM[i] - steps_total_RM[i] % (y + 1)) / (y + 1);
			if (NODE[i] >= rem){
				if (tt == -1){
					fp = NODE[i];
					ip = steps_total_RM[i] - steps_total_RM[i] % (y + 1);
					tt = 0;
				}
				NODE[i] = fp + ((steps_total_RM[i] - ip) - (steps_total_RM[i] - ip) % y) / y;
			}
			if (NODE[i] >= numtasks) fprintf(stderr, "Error balance_load() \n");
			if (NODE[i] == rank){
				n_active_local_max++;
				n_active_local_min = n_active_local_max;
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_max, 1, MPI_INT, MPI_MAX, mpi_cartcomm);
		MPI_Allreduce(MPI_IN_PLACE, &n_active_local_min, 1, MPI_INT, MPI_MIN, mpi_cartcomm);
	} while ((n_active_local_max > MAX_BLOCKS || (n_active_local_min == 0 && n_active_total > numtasks)) && timelevel_cutoff >= 2);
	if (rank == 0 && timelevel_cutoff != AMR_MAXTIMELEVEL) fprintf(stderr, "Error in balance_load. Due to too little/many blocks the maximum timelevel can't be honoured and the hierarchical timestepping is downgraded! \n");
	if (rank == 0 && (n_active_local_max > MAX_BLOCKS)) fprintf(stderr, "Error in balance_load: Too many blocks refined, possible to get OpenCL or OOM errors! \n");
	if (rank == 0) fprintf(stderr, "Load balance started, timelevel_cutoff %d %d %d! \n", timelevel_cutoff, n_active_local_min, n_active_local_max);

	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[i]){
			if (block[n_ord_total_RM[i]][AMR_NODE] == rank){
				rc = MPI_Isend(&p[n_ord_total_RM[i]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[i], (2 * n_ord_total_RM[i] + 0) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_ord_total_RM[i]][0]);
				#if STAGGERED
				rc += MPI_Isend(&ps[n_ord_total_RM[i]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, NODE[i], (2 * n_ord_total_RM[i] + 1) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_ord_total_RM[i]][1]);
				#endif
				if (rc != 0)fprintf(stderr, "Error balance_load send %d", rc);
			}
		}
	}
	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[i]){
			if (NODE[i] == rank){
				//Allocate memory for active blocks on node
				set_arrays(n_ord_total_RM[i]);
				set_grid(n_ord_total_RM[i]);
				if (block[n_ord_total_RM[i]][AMR_NODE] >= 0){
					rc = MPI_Irecv(&p[n_ord_total_RM[i]][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (2 * n_ord_total_RM[i] + 0) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_ord_total_RM[i]][10]);
					#if STAGGERED
					rc += MPI_Irecv(&ps[n_ord_total_RM[i]][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_ord_total_RM[i]][AMR_NODE], (2 * n_ord_total_RM[i] + 1) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_ord_total_RM[i]][11]);
					#endif
					if (rc != 0)fprintf(stderr, "Error balance_load receive %d", rc);
				}
			}
		}
	}

	for (i = 0; i < n_active_total; i++){

		//Then use MPI_wait to clean up data that has been sent
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[i]){
			if (block[n_ord_total_RM[i]][AMR_NODE] == rank){
				MPI_Wait(&boundreqs[n_ord_total_RM[i]][0], &Statbound[n_ord_total_RM[i]][0]);
				#if STAGGERED
				MPI_Wait(&boundreqs[n_ord_total_RM[i]][1], &Statbound[n_ord_total_RM[i]][1]);
				#endif
				#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
				GPU_finish(n_ord_total_RM[i]);
				#endif
				free_arrays(n_ord_total_RM[i]);
			}
		}
	}

	//Then use MPI_wait to receive data 
	for (i = 0; i < n_active_total; i++){
		if (block[n_ord_total_RM[i]][AMR_NODE] != NODE[i]){
			if (NODE[i] == rank){
				if (block[n_ord_total_RM[i]][AMR_NODE] >= 0){
					MPI_Wait(&boundreqs[n_ord_total_RM[i]][10], &Statbound[n_ord_total_RM[i]][10]);
					#if STAGGERED
					MPI_Wait(&boundreqs[n_ord_total_RM[i]][11], &Statbound[n_ord_total_RM[i]][11]);
					#endif
				}
				#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
				set_arrays_GPU(n_ord_total_RM[i], block[n_ord_total_RM[i]][AMR_GPU]);
				GPU_write(n_ord_total_RM[i]);
				#endif
			}
		}
	}

	for (i = 0; i < n_active_total; i++){
		block[n_ord_total_RM[i]][AMR_NODE] = NODE[i];
	}

	activate_blocks();
	count_node[0]=0;
	for (n = 0; n < n_active; n++){
		count_node[0] += AMR_MAXTIMELEVEL / block[n_ord_RM[n]][AMR_TIMELEVEL];
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
	GPU_boundprim(1);

	if (rank == 0) fprintf(stderr, "Load balance finished! \n");
}


void balance_load_gpu(void){
	int i, g, tt, fp, ip, y, q, rem, n, gpu;
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
		steps_RM[n] = n_active_steps + AMR_MAXTIMELEVEL / 2 / block[n_ord_RM[n]][AMR_TIMELEVEL];
		n_active_steps += AMR_MAXTIMELEVEL / block[n_ord_RM[n]][AMR_TIMELEVEL];
	}

	tt = -1, ip = 0, fp = 0, rem, gpu = 0;
	rem = n_active_steps % (NQ); //remainder of last unfilled block
	y = (n_active_steps - rem) / (NQ); //number of blocks/gpu

	for (n = 0; n < n_active; n++){
		gpu = (steps_RM[n] - steps_RM[n] % (y + 1)) / (y + 1);
		if (gpu >= rem){
			if (tt == -1){
				fp = gpu;
				ip = steps_RM[n] - steps_RM[n] % (y + 1);
				tt = 0;
			}
			gpu = fp + ((steps_RM[n] - ip) - (steps_RM[n] - ip) % y) / y;
		}
		if (gpu >= NQ) fprintf(stderr, "Error balance_load_gpu() \n");
		commandQueueGPU[n_ord_RM[n]] = commandQueue[gpu];
	}
}

/*Function calculates the ordered arrays of all active blocks on a single node (n_active) and on the whole cluster (n_active_total) */
void activate_blocks(void){
	int n, i;

	n_active = 0;
	n_active_total = 0;
	
	for (n = 0; n <= n_max; n++){
		flux_flag[n] = 0;


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
			n_active_total++;
			if (block[n][AMR_LEVEL] > 0) block[block[n][AMR_PARENT]][AMR_REFINED] = 1;
			block[n][AMR_REFINED] = 0;
		}
	}

	//Set the same array for each GPU on a single node
	for (i = 0; i < N_GPU; i++){
		n_active_gpu[i] = 0;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_GPU] == i){
				n_ord_gpu[i][n_active_gpu[i]] = n;
				n_active_gpu[i]++;
			}
		}
	}
}

int rm_order2(void){
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

			if (block[n_ord_total[n]][AMR_NBR1] >= 0 && block[n_ord_total[n]][AMR_POLE] != 1 && block[n_ord_total[n]][AMR_POLE] != 4){
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
					counter1_1 -= block[block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD7]][AMR_TIMELEVEL];
					counter1_2 -= block[block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD7];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD8];
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
					counter2_1 -= block[block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD2]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN1D_1] = block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN1D_2] = block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD2];
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
					counter3_1 -= block[block[block[n_ord_total[n]][AMR_CORN1]][AMR_CHILD3]][AMR_TIMELEVEL];
					counter3_2 -= block[block[block[n_ord_total[n]][AMR_CORN1]][AMR_CHILD4]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN1D_1] = block[block[n_ord_total[n]][AMR_CORN1]][AMR_CHILD3];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN1D_2] = block[block[n_ord_total[n]][AMR_CORN1]][AMR_CHILD4];
				}
			}
		}
	}

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
					block[n_ord_total[n]][AMR_CORN2D_1] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_CHILD5]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN2D_2] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_CHILD6]][AMR_CORN1D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN2]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN2]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN2D] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN2D] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_CORN1D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN2]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN2])
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
					block[n_ord_total[n]][AMR_CORN3D_1] = block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_CHILD5]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN3D_2] = block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_CHILD6]][AMR_CORN1D];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN3]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN3D] = block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN3D] = block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT]][AMR_CORN1D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN3]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN3])
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
					block[n_ord_total[n]][AMR_CORN4D_1] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_CHILD5]][AMR_CORN1D];
					block[n_ord_total[n]][AMR_CORN4D_2] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_CHILD6]][AMR_CORN1D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN4]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN4]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD7]){
						block[n_ord_total[n]][AMR_CORN4D] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_CORN1D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN4D] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_CORN1D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN4]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN4])
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
					counter1_1 -= block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD6]][AMR_TIMELEVEL];
					counter1_2 -= block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD6];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD8];
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
					counter2_1 -= block[block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD3]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN5D_1] = block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN5D_2] = block[block[n_ord_total[n]][AMR_NBR2]][AMR_CHILD3];
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
					counter3_1 -= block[block[block[n_ord_total[n]][AMR_CORN5]][AMR_CHILD2]][AMR_TIMELEVEL];
					counter3_2 -= block[block[block[n_ord_total[n]][AMR_CORN5]][AMR_CHILD4]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN5D_1] = block[block[n_ord_total[n]][AMR_CORN5]][AMR_CHILD2];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN5D_2] = block[block[n_ord_total[n]][AMR_CORN5]][AMR_CHILD4];
				}
			}
		}
	}

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
					block[n_ord_total[n]][AMR_CORN6D_1] = block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN6D_2] = block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD7]][AMR_CORN5D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN6]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN6]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN6D] = block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN6D] = block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT]][AMR_CORN5D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN6]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN6])
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
					block[n_ord_total[n]][AMR_CORN7D_1] = block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_CHILD5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN7D_2] = block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_CHILD7]][AMR_CORN5D];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN7]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN7D] = block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN7D] = block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT]][AMR_CORN5D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN7]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN7])
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
					block[n_ord_total[n]][AMR_CORN8D_1] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_CHILD5]][AMR_CORN5D];
					block[n_ord_total[n]][AMR_CORN8D_2] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_CHILD7]][AMR_CORN5D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN8]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN8]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD5] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD6]){
						block[n_ord_total[n]][AMR_CORN8D] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_CORN5D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN8D] = block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_CORN5D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR4]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN8]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN8])
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
					counter1_1 -= block[block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD4]][AMR_TIMELEVEL];
					counter1_2 -= block[block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD8]][AMR_TIMELEVEL];

					if (counter1_1 > counter0_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD4];
					if (counter1_2 > counter0_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[block[n_ord_total[n]][AMR_NBR1]][AMR_CHILD8];
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
					counter2_1 -= block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD1]][AMR_TIMELEVEL];
					counter2_2 -= block[block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD5]][AMR_TIMELEVEL];
					if (counter2_1 > counter1_1) block[n_ord_total[n]][AMR_CORN9D_1] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD1];
					if (counter2_2 > counter1_2) block[n_ord_total[n]][AMR_CORN9D_2] = block[block[n_ord_total[n]][AMR_NBR5]][AMR_CHILD5];
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
					counter3_1 -= block[block[block[n_ord_total[n]][AMR_CORN9]][AMR_CHILD3]][AMR_TIMELEVEL];
					counter3_2 -= block[block[block[n_ord_total[n]][AMR_CORN9]][AMR_CHILD7]][AMR_TIMELEVEL];

					if ((counter3_1 > counter2_1) && (counter3_1>counter1_1)) block[n_ord_total[n]][AMR_CORN9D_1] = block[block[n_ord_total[n]][AMR_CORN9]][AMR_CHILD3];
					if ((counter3_2 > counter2_2) && (counter3_2>counter1_2)) block[n_ord_total[n]][AMR_CORN9D_2] = block[block[n_ord_total[n]][AMR_CORN9]][AMR_CHILD7];
				}
			}
		}
	}

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
					block[n_ord_total[n]][AMR_CORN10D_1] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_CHILD2]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN10D_2] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_CHILD6]][AMR_CORN9D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN10]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN10]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN10D] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN10D] = block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_CORN9D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR3]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN10]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN10])
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
					block[n_ord_total[n]][AMR_CORN11D_1] = block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_CHILD2]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN11D_2] = block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_CHILD6]][AMR_CORN9D];
				}
				else if (block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN11]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN11D] = block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN11D] = block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT]][AMR_CORN9D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN11]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN11])
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
					block[n_ord_total[n]][AMR_CORN12D_1] = block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD2]][AMR_CORN9D];
					block[n_ord_total[n]][AMR_CORN12D_2] = block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_CHILD6]][AMR_CORN9D];
				}
				else if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN12]][AMR_PARENT] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN12]){
					if (n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD2] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD3] || n_ord_total[n] == block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD4]){
						block[n_ord_total[n]][AMR_CORN12D] = block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT]][AMR_CORN9D_1];
					}
					else{
						block[n_ord_total[n]][AMR_CORN12D] = block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT]][AMR_CORN9D_2];
					}
				}
				else if (block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT] >= 0 && block[block[block[n_ord_total[n]][AMR_NBR6]][AMR_PARENT]][AMR_ACTIVE] == 1 && block[block[n_ord_total[n]][AMR_CORN12]][AMR_PARENT] != block[block[n_ord_total[n]][AMR_PARENT]][AMR_CORN12])
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
						p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 0.125/gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][CENT] * (
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child], (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child])][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child], (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][CENT] +
							p[n_child][index(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][k] * gdet[n][index2(n_child, (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child] + REF_1, (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child] + REF_2, (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child] + REF_3)][CENT]);
					}
				}
			}
		}

		//Average conserved quantitites
		#pragma omp for collapse(2) schedule(dynamic)
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
						get_state(p[n_child][index(n_child, i_1, j_1, z_1)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_1, j_1, z_1)], &q, &geom, dq[n_child][index(n_child, i_1, j_1, z_1)]);

						get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
						get_state(p[n_child][index(n_child, i_1, j_1, z_2)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_1, j_1, z_2)], &q, &geom, dq[n_child][index(n_child, i_1, j_1, z_2)]);
						
						get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
						get_state(p[n_child][index(n_child, i_1, j_2, z_1)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_1, j_2, z_1)], &q, &geom, dq[n_child][index(n_child, i_1, j_2, z_1)]);

						get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
						get_state(p[n_child][index(n_child, i_1, j_2, z_2)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_1, j_2, z_2)], &q, &geom, dq[n_child][index(n_child, i_1, j_2, z_2)]);
						
						get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
						get_state(p[n_child][index(n_child, i_2, j_1, z_1)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_2, j_1, z_1)], &q, &geom, dq[n_child][index(n_child, i_2, j_1, z_1)]);
						
						get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
						get_state(p[n_child][index(n_child, i_2, j_1, z_2)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_2, j_1, z_2)], &q, &geom, dq[n_child][index(n_child, i_2, j_1, z_2)]);
						
						get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
						get_state(p[n_child][index(n_child, i_2, j_2, z_1)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_2, j_2, z_1)], &q, &geom, dq[n_child][index(n_child, i_2, j_2, z_1)]);
						
						get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
						get_state(p[n_child][index(n_child, i_2, j_2, z_2)], &geom, &q);
						primtoU(p[n_child][index(n_child, i_2, j_2, z_2)], &q, &geom, dq[n_child][index(n_child, i_2, j_2, z_2)]);

						dq[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
							= 0.125  * (
							dq[n_child][index(n_child, i_1, j_1, z_1)][k] +
							dq[n_child][index(n_child, i_1, j_1, z_2)][k] +
							dq[n_child][index(n_child, i_1, j_2, z_1)][k] +
							dq[n_child][index(n_child, i_1, j_2, z_2)][k] +
							dq[n_child][index(n_child, i_2, j_1, z_1)][k] +
							dq[n_child][index(n_child, i_2, j_1, z_2)][k] +
							dq[n_child][index(n_child, i_2, j_2, z_1)][k] +
							dq[n_child][index(n_child, i_2, j_2, z_2)][k] );
					}
					pflag[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])] = Utoprim_2d(dq[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])], geom.gcov, geom.gcon, geom.g, p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])]);
				}
			}
		}
		#if STAGGERED
		#pragma omp for collapse(2) schedule(dynamic)
		for (i = i1; i < i2 + D1; i++){
			for (j = j1; j < j2; j++){
				for (z = z1; z < z2; z++){
					ic = (i - i1)*(1 + REF_1) + N1_GPU_offset[n_child];
					jc = (j - j1)*(1 + REF_2) + N2_GPU_offset[n_child];
					zc = (z - z1)*(1 + REF_3) + N3_GPU_offset[n_child];
					k = 1;
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE1] * 0.25*(
						ps[n_child][index(n_child, ic, jc, zc)][k] * gdet[n_child][index2(n_child, ic, jc, zc)][FACE1] +
						ps[n_child][index(n_child, ic, jc + REF_2, zc)][k] * gdet[n_child][index2(n_child, ic, jc + REF_2, zc)][FACE1] +
						ps[n_child][index(n_child, ic, jc, zc + REF_3)][k] * gdet[n_child][index2(n_child, ic, jc, zc + REF_3)][FACE1] +
						ps[n_child][index(n_child, ic, jc + REF_2, zc + REF_3)][k] * gdet[n_child][index2(n_child, ic, jc + REF_2, zc + REF_3)][FACE1]);
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
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE2] * 0.25*(
						ps[n_child][index(n_child, ic, jc, zc)][k] * gdet[n_child][index2(n_child, ic, jc, zc)][FACE2] +
						ps[n_child][index(n_child, ic + REF_1, jc, zc)][k] * gdet[n_child][index2(n_child, ic + REF_1, jc, zc)][FACE2] +
						ps[n_child][index(n_child, ic, jc, zc + REF_3)][k] * gdet[n_child][index2(n_child, ic, jc, zc + REF_3)][FACE2] +
						ps[n_child][index(n_child, ic + REF_1, jc, zc + REF_3)][k] * gdet[n_child][index2(n_child, ic + REF_1, jc, zc + REF_3)][FACE2]);
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
					ps[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]
						= 1. / gdet[n][index2(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][FACE3] * 0.25*(
						ps[n_child][index(n_child, ic, jc, zc)][k] * gdet[n_child][index2(n_child, ic, jc, zc)][FACE3] +
						ps[n_child][index(n_child, ic + REF_1, jc, zc)][k] * gdet[n_child][index2(n_child, ic + REF_1, jc, zc)][FACE3] +
						ps[n_child][index(n_child, ic, jc + REF_2, zc)][k] * gdet[n_child][index2(n_child, ic, jc + REF_2, zc)][FACE3] +
						ps[n_child][index(n_child, ic + REF_1, jc + REF_2, zc)][k] * gdet[n_child][index2(n_child, ic + REF_1, jc + REF_2, zc)][FACE3]);
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
			block_average(n, n_child, 0, N1_GPU[n_child] / (1 + REF_1), 0, N2_GPU[n_child] / (1 + REF_2), 0, N3_GPU[n_child] / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD1]);
			GPU_finish(block[n][AMR_CHILD1]);
			#endif
		}

		if (block[n][AMR_CHILD2] >= 0 && REF_3 == 1){
			n_child = block[n][AMR_CHILD2];
			block_average(n, n_child, 0, N1_GPU[n_child] / (1 + REF_1), 0, N2_GPU[n_child] / (1 + REF_2), N3_GPU[n_child] / (1 + REF_3), N3_GPU[n_child]);
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD2]);
			GPU_finish(block[n][AMR_CHILD2]);
			#endif
		}

		if (block[n][AMR_CHILD3] >= 0 && REF_2 == 1){
			n_child = block[n][AMR_CHILD3];
			block_average(n, n_child, 0, N1_GPU[n_child] / (1 + REF_1), N2_GPU[n_child] / (1 + REF_2), N2_GPU[n_child], 0, N3_GPU[n_child] / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD3]);
			GPU_finish(block[n][AMR_CHILD3]);
			#endif
		}

		if (block[n][AMR_CHILD4] >= 0 && REF_2 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD4];
			block_average(n, n_child, 0, N1_GPU[n_child] / (1 + REF_1), N2_GPU[n_child] / (1 + REF_2), N2_GPU[n_child], N3_GPU[n_child] / (1 + REF_3), N3_GPU[n_child]);
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD4]);
			GPU_finish(block[n][AMR_CHILD4]);
			#endif
		}

		if (block[n][AMR_CHILD5] >= 0 && REF_1 == 1){
			n_child = block[n][AMR_CHILD5];
			block_average(n, n_child, N1_GPU[n_child] / (1 + REF_1), N1_GPU[n_child], 0, N2_GPU[n_child] / (1 + REF_2), 0, N3_GPU[n_child] / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD5]);
			GPU_finish(block[n][AMR_CHILD5]);
			#endif
		}

		if (block[n][AMR_CHILD6] >= 0 && REF_1 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD6];
			block_average(n, n_child, N1_GPU[n_child] / (1 + REF_1), N1_GPU[n_child], 0, N2_GPU[n_child] / (1 + REF_2), N3_GPU[n_child] / (1 + REF_3), N3_GPU[n_child]);
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD6]);
			GPU_finish(block[n][AMR_CHILD6]);
			#endif
		}

		if (block[n][AMR_CHILD7] >= 0 && REF_1 == 1 && REF_2 == 1){
			n_child = block[n][AMR_CHILD7];
			block_average(n, n_child, N1_GPU[n_child] / (1 + REF_1), N1_GPU[n_child], N2_GPU[n_child] / (1 + REF_2), N2_GPU[n_child], 0, N3_GPU[n_child] / (1 + REF_3));
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			free_arrays(block[n][AMR_CHILD7]);
			GPU_finish(block[n][AMR_CHILD7]);
			#endif
		}

		if (block[n][AMR_CHILD8] >= 0 && REF_1 == 1 && REF_2 == 1 && REF_3 == 1){
			n_child = block[n][AMR_CHILD8];
			block_average(n, n_child, N1_GPU[n_child] / (1 + REF_1), N1_GPU[n_child], N2_GPU[n_child] / (1 + REF_2), N2_GPU[n_child], N3_GPU[n_child] / (1 + REF_3), N3_GPU[n_child]);
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
	
	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	if (block[n][AMR_NODE] == rank){
		set_arrays_GPU(n, block[n][AMR_GPU]);
		GPU_write(n);
	}
	#endif
}

void refine_cell(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict prim[NB])[NPR], double(*restrict d1[NB])[NPR], double(*restrict d2[NB])[NPR], double(*restrict d3[NB])[NPR]){
	int i, j, z, i1, j1, z1, k, i_1, j_1, z_1, i_2, j_2, z_2;
	double U[NPR], U0[NPR], factor[NPR];
	struct of_geom geom;
	struct of_state q;
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, i_1, j_1, z_1, i_2, j_2, z_2,geom,q,U,U0,factor)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, N1_GPU[n_child] - 1, 0, N2_GPU[n_child] - 1, 0, N3_GPU[n_child] - 1) {
			i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*N1_GPU[n] / 2 * REF_1;
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*N2_GPU[n] / 2 * REF_2;
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*N3_GPU[n] / 2 * REF_3;
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] - 0.25 * d1[n][index(n, i1, j1, z1)][k] - 0.25 * d2[n][index(n, i1, j1, z1)][k] - 0.25 * d3[n][index(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] - 0.25 * d1[n][index(n, i1, j1, z1)][k] - 0.25 * d2[n][index(n, i1, j1, z1)][k] + 0.25 * d3[n][index(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] - 0.25 * d1[n][index(n, i1, j1, z1)][k] + 0.25 * d2[n][index(n, i1, j1 + REF_2, z1)][k] - 0.25 * d3[n][index(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] - 0.25 * d1[n][index(n, i1, j1, z1)][k] + 0.25 * d2[n][index(n, i1, j1 + REF_2, z1)][k] + 0.25 * d3[n][index(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] + 0.25 * d1[n][index(n, i1 + REF_1, j1, z1)][k] - 0.25 * d2[n][index(n, i1, j1, z1)][k] - 0.25 * d3[n][index(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] + 0.25 * d1[n][index(n, i1 + REF_1, j1, z1)][k] - 0.25 * d2[n][index(n, i1, j1, z1)][k] + 0.25 * d3[n][index(n, i1, j1, z1 + REF_3)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] + 0.25 * d1[n][index(n, i1 + REF_1, j1, z1)][k] + 0.25 * d2[n][index(n, i1, j1 + REF_2, z1)][k] - 0.25 * d3[n][index(n, i1, j1, z1)][k];
				}
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				PLOOP{
					prim[n_child][index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child])][k] =
					prim[n][index(n, i1, j1, z1)][k] + 0.25 * d1[n][index(n, i1 + REF_1, j1, z1)][k] + 0.25 * d2[n][index(n, i1, j1 + REF_2, z1)][k] + 0.25 * d3[n][index(n, i1, j1, z1 + REF_3)][k];
				}
			}
			//Enforce strict conservation of conservative quantitites during refinement
			/*if (i % (1 + REF_1) == REF_1 && j % (1 + REF_2) == REF_2 && z % (1 + REF_3) == REF_3){
				i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*N1_GPU[n] / 2 * REF_1;
				j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*N2_GPU[n] / 2 * REF_2;
				z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*N3_GPU[n] / 2 * REF_3;

				i_1 = N1_GPU_offset[n_child] + i - REF_1;
				i_2 = N1_GPU_offset[n_child] + i;
				j_1 = N2_GPU_offset[n_child] + j - REF_2;
				j_2 = N2_GPU_offset[n_child] + j;
				z_1 = N3_GPU_offset[n_child] + z - REF_3;
				z_2 = N3_GPU_offset[n_child] + z;

				//Calculate total conserved quantities in new cells
				get_geometry(n_child, i_1, j_1, z_1, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_1, j_1, z_1)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_1, j_1, z_1)], &q, &geom, dq[n_child][index(n_child, i_1, j_1, z_1)]);
				PLOOP U[k] = 0.125*(dq[n_child][index(n_child, i_1, j_1, z_1)][k]);

				get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_1, j_1, z_2)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_1, j_1, z_2)], &q, &geom, dq[n_child][index(n_child, i_1, j_1, z_2)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_1, j_1, z_2)][k]);

				get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_1, j_2, z_1)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_1, j_2, z_1)], &q, &geom, dq[n_child][index(n_child, i_1, j_2, z_1)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_1, j_2, z_1)][k]);

				get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_1, j_2, z_2)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_1, j_2, z_2)], &q, &geom, dq[n_child][index(n_child, i_1, j_2, z_2)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_1, j_2, z_2)][k]);

				get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_2, j_1, z_1)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_2, j_1, z_1)], &q, &geom, dq[n_child][index(n_child, i_2, j_1, z_1)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_2, j_1, z_1)][k]);

				get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_2, j_1, z_2)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_2, j_1, z_2)], &q, &geom, dq[n_child][index(n_child, i_2, j_1, z_2)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_2, j_1, z_2)][k]);

				get_geometry(n_child, i_2, j_2, z_1, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_2, j_2, z_1)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_2, j_2, z_1)], &q, &geom, dq[n_child][index(n_child, i_2, j_2, z_1)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_2, j_2, z_1)][k]);

				get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
				get_state(prim[n_child][index(n_child, i_2, j_2, z_2)], &geom, &q);
				primtoU(prim[n_child][index(n_child, i_2, j_2, z_2)], &q, &geom, dq[n_child][index(n_child, i_2, j_2, z_2)]);
				PLOOP U[k] += 0.125*(dq[n_child][index(n_child, i_2, j_2, z_2)][k]);

				//Calculate conserved quantities in old cell
				get_geometry(n, i1, j1, z1, CENT, &geom);
				get_state(prim[n][index(n, i1, j1, z1)], &geom, &q);
				primtoU(prim[n][index(n, i1, j1, z1)], &q, &geom, dq[n][index(n, i1, j1, z1)]);
				PLOOP U0[k] = (dq[n][index(n, i1, j1, z1)][k]);

				//Normalize new conserved quantities in children to parent cell and convert back to primitive variables
				PLOOP factor[k] = U0[k] / U[k];
				
				PLOOP dq[n_child][index(n_child, i_1, j_1, z_1)][k] *= factor[k];
				get_geometry(n_child, i_1, j_1, z_1, CENT, &geom);
				pflag[n_child][index(n_child, i_1, j_1, z_1)] = Utoprim_2d(dq[n_child][index(n_child, i_1, j_1, z_1)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_1, j_1, z_1)]);
				if (REF_3 == 1){
					PLOOP dq[n_child][index(n_child, i_1, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_1, z_2, CENT, &geom);
					pflag[n_child][index(n_child, i_1, j_1, z_2)]=Utoprim_2d(dq[n_child][index(n_child, i_1, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_1, j_1, z_2)]);
				}
				if (REF_2 == 1){
					PLOOP dq[n_child][index(n_child, i_1, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_1, CENT, &geom);
					pflag[n_child][index(n_child, i_1, j_1, z_2)] = Utoprim_2d(dq[n_child][index(n_child, i_1, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_1, j_2, z_1)]);
				}
				if (REF_2==1 && REF_3 == 1){
					PLOOP dq[n_child][index(n_child, i_1, j_2, z_2)][k] *= factor[k];
					get_geometry(n_child, i_1, j_2, z_2, CENT, &geom);
					pflag[n_child][index(n_child, i_1, j_2, z_2)] = Utoprim_2d(dq[n_child][index(n_child, i_1, j_2, z_2)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_1, j_2, z_2)]);
				}
				if (REF_1 == 1){
					PLOOP dq[n_child][index(n_child, i_2, j_1, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_1, CENT, &geom);
					pflag[n_child][index(n_child, i_2, j_1, z_1)] = Utoprim_2d(dq[n_child][index(n_child, i_2, j_1, z_1)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_2, j_1, z_1)]);
				}
				if (REF_1 == 1 && REF_3==1){
					PLOOP dq[n_child][index(n_child, i_2, j_1, z_2)][k] *= factor[k];
					get_geometry(n_child, i_2, j_1, z_2, CENT, &geom);
					pflag[n_child][index(n_child, i_2, j_1, z_2)] = Utoprim_2d(dq[n_child][index(n_child, i_2, j_1, z_2)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_2, j_1, z_2)]);
				}
				if (REF_1 == 1 && REF_2 == 1){
					PLOOP dq[n_child][index(n_child, i_2, j_2, z_1)][k] *= factor[k];
					get_geometry(n_child, i_2, j_2, z_1, CENT, &geom);
					pflag[n_child][index(n_child, i_2, j_2, z_1)] = Utoprim_2d(dq[n_child][index(n_child, i_2, j_2, z_1)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_2, j_2, z_1)]);
				}
				if (REF_1 == 1 && REF_2 == 1 && REF_3 == 1){
					PLOOP dq[n_child][index(n_child, i_2, j_2, z_2)][k] *= factor[k];
					get_geometry(n_child, i_2, j_2, z_2, CENT, &geom);
					pflag[n_child][index(n_child, i_2, j_2, z_2)] = Utoprim_2d(dq[n_child][index(n_child, i_2, j_2, z_2)], geom.gcov, geom.gcon, geom.g, prim[n_child][index(n_child, i_2, j_2, z_2)]);
				}
			}*/
		}
	}
}


void refine_field(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict pb[NB])[NDIM]){
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

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_5[n]; n_rec4 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_6[n]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_7[n]; n_rec4 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_REFINED] == 1){ pointer4 = receive4_8[n]; n_rec4 = 1; }

	if (offset_2 == 0 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_1[n]; n_rec2 = 1; }
	if (offset_2 == 0 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_2[n]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 0 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_3[n]; n_rec2 = 1; }
	if (offset_2 == 1 && offset_3 == 1 && block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_REFINED] == 1){ pointer2 = receive2_4[n]; n_rec2 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_1[n]; n_rec3 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_2[n]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_5[n]; n_rec3 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_REFINED] == 1){ pointer3 = receive3_6[n]; n_rec3 = 1; }

	if (offset_1 == 0 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_3[n]; n_rec1 = 1; }
	if (offset_1 == 0 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_4[n]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 0 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_7[n]; n_rec1 = 1; }
	if (offset_1 == 1 && offset_3 == 1 && block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_REFINED] == 1){ pointer1 = receive1_8[n]; n_rec1 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_1[n]; n_rec5 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_3[n]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_5[n]; n_rec5 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_REFINED] == 1){ pointer5 = receive5_7[n]; n_rec5 = 1; }

	if (offset_1 == 0 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_2[n]; n_rec6 = 1; }
	if (offset_1 == 0 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_4[n]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 0 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_6[n]; n_rec6 = 1; }
	if (offset_1 == 1 && offset_2 == 1 && block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_REFINED] == 1){ pointer6 = receive6_8[n]; n_rec6 = 1; }
	
	isize = N1_GPU[n_child];
	jsize = N2_GPU[n_child];
	zsize = N3_GPU[n_child];
	#pragma omp parallel private(i, j, z, i1, j1, z1, k, ind0, ind2,b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8,b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8,set_1, set_2,set_3,set_4,set_5,set_6)
	{
	#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(0, N1_GPU[n_child] - 1 + D1, 0, N2_GPU[n_child] - 1 + D2, 0, N3_GPU[n_child] - 1 + D3) {
			//indices in coarse grid depending on offset (input parameter)
			i1 = (i - i % (1 + REF_1)) / (1 + REF_1);
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2);
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3);

			//index of child
			ind0 = index(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);
			ind2 = index2(n_child, i + N1_GPU_offset[n_child], j + N2_GPU_offset[n_child], z + N3_GPU_offset[n_child]);

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
			if ((i == N1_GPU[n_child] || i == N1_GPU[n_child] - REF_1 || i == N1_GPU[n_child] - (1 + REF_1)) && (offset_1 == 1 || REF_1 == 0) && n_rec4 >= 0){
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
			if ((j == N2_GPU[n_child] || j == N2_GPU[n_child] - REF_2 || j == N2_GPU[n_child] - (1 + REF_2)) && (offset_2 == 1 || REF_2 == 0) && n_rec1 >= 0){
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
			if ((z == N3_GPU[n_child] || z == N3_GPU[n_child] - REF_3 || z == N3_GPU[n_child] - (1 + REF_3)) && (offset_3 == 1 || REF_3 == 0) && n_rec6 >= 0){
				b3_2 = pointer6[i1*(1 + REF_1)*jsize + j1*(1 + REF_2)];
				b3_4 = pointer6[i1*(1 + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				b3_6 = pointer6[(i1*(1 + REF_1) + REF_1)*jsize + j1*(1 + REF_2)];
				b3_8 = pointer6[(i1*(1 + REF_1) + REF_1)*jsize + (j1*(1 + REF_2) + REF_2)];
				set_6 = 1;
			}
			else set_6 = -1;

			i1 = (i - i % (1 + REF_1)) / (1 + REF_1) + N1_GPU_offset[n] + offset_1*N1_GPU[n] / 2 * REF_1 - (i == N1_GPU[n] && (offset_1 == 1 || REF_1 == 0));
			j1 = (j - j % (1 + REF_2)) / (1 + REF_2) + N2_GPU_offset[n] + offset_2*N2_GPU[n] / 2 * REF_2 - (j == N2_GPU[n] && (offset_2 == 1 || REF_2 == 0));
			z1 = (z - z % (1 + REF_3)) / (1 + REF_3) + N3_GPU_offset[n] + offset_3*N3_GPU[n] / 2 * REF_3 - (z == N3_GPU[n] && (offset_3 == 1 || REF_3 == 0));

			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == N1_GPU[n] && (offset_1 == 1 || REF_1 == 0)), -0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, -0.5 + (double)(j == N2_GPU[n] && (offset_2 == 1 || REF_2 == 0)), -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, -0.25*REF_2, -0.5 + (double)(z == N3_GPU[n] && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}

			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == N1_GPU[n] && (offset_1 == 1 || REF_1 == 0)), -0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, -0.5 + (double)(j == N2_GPU[n] && (offset_2 == 1 || REF_2 == 0)), 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, -0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == N1_GPU[n] && (offset_1 == 1 || REF_1 == 0)), 0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, 0.0, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, 0.25*REF_2, -0.5 + (double)(z == N3_GPU[n] && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 0 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, -0.5 + (double)(i == N1_GPU[n] && (offset_1 == 1 || REF_1 == 0)), 0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, -0.25*REF_1, 0.0, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, -0.25*REF_1, 0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 0){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, -0.5 + (double)(j == N2_GPU[n] && (offset_2 == 1 || REF_2 == 0)), -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, -0.25*REF_2, -0.5 + (double)(z == N3_GPU[n] && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 0 && z % (1 + REF_3) == 1){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, -0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, -0.5 + (double)(j == N2_GPU[n] && (offset_2 == 1 || REF_2 == 0)), 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, -0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 0){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*REF_2, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, 0.0, -0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, 0.25*REF_2, -0.5 + (double)(z == N3_GPU[n] && (offset_3 == 1 || REF_3 == 0)), psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}
			if (i % (1 + REF_1) == 1 && j % (1 + REF_2) == 1 && z % (1 + REF_3) == 1){
				pb[n_child][ind0][1] =
					1. / gdet[n_child][ind2][FACE1] * B1_prolong(n, i1, j1, z1, 0.0, 0.25*REF_2, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][2] =
					1. / gdet[n_child][ind2][FACE2] * B2_prolong(n, i1, j1, z1, 0.25*REF_1, 0.0, 0.25*REF_3, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
				pb[n_child][ind0][3] =
					1. / gdet[n_child][ind2][FACE3] * B3_prolong(n, i1, j1, z1, 0.25*REF_1, 0.25*REF_2, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
					b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, set_1, set_2, set_3, set_4, set_5, set_6);
			}


			#if (WHICHPROBLEM==DISRUPTION_PROBLEM)
			pb[n_child][ind0][1] = 0.;
			pb[n_child][ind0][2] = 0.;
			pb[n_child][ind0][3] = 0.;
			#endif
		}
	}
}

void pre_refine(void){
	int n1, i, j, z;
	for (n1 = 0; n1 < n_active; n1++){
		#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
		GPU_read(n_ord[n1]);
		#endif
		#pragma omp parallel private(i, j, z)
		{
			#pragma omp for collapse(2) schedule(dynamic)
			ZSLOOP3D(N1_GPU_offset[n_ord[n1]] - N1G, N1_GPU_offset[n_ord[n1]] + N1_GPU[n_ord[n1]] + N1G - 1, -N2G + N2_GPU_offset[n_ord[n1]], N2_GPU_offset[n_ord[n1]] + N2_GPU[n_ord[n1]] + N2G - 1, N3_GPU_offset[n_ord[n1]] - N3G, N3_GPU_offset[n_ord[n1]] + N3_GPU[n_ord[n1]] + N3G - 1) {
				psh[n_ord[n1]][index(n_ord[n1], i, j, z)][1] = ps[n_ord[n1]][index(n_ord[n1], i, j, z)][1] * gdet[n_ord[n1]][index2(n_ord[n1], i, j, z)][FACE1];
				psh[n_ord[n1]][index(n_ord[n1], i, j, z)][2] = ps[n_ord[n1]][index(n_ord[n1], i, j, z)][2] * gdet[n_ord[n1]][index2(n_ord[n1], i, j, z)][FACE2];
				psh[n_ord[n1]][index(n_ord[n1], i, j, z)][3] = ps[n_ord[n1]][index(n_ord[n1], i, j, z)][3] * gdet[n_ord[n1]][index2(n_ord[n1], i, j, z)][FACE3];
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
			ZSLOOP3D(-D1, N1_GPU[n] - 1 + D1, -D2, N2_GPU[n] - 1 + D2, -D3, N3_GPU[n] - 1 + D3) {
				PLOOP{
					F1[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[n][index(n, i + N1_GPU_offset[n] - D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[n][index(n, i + N1_GPU_offset[n] + D1, j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k]);
					F2[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] - D2, z + N3_GPU_offset[n])][k], p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n] + D2, z + N3_GPU_offset[n])][k]);
					#if(N3>1)
					F3[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k] = slope_lim(p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] - D3)][k], p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n])][k], p[n][index(n, i + N1_GPU_offset[n], j + N2_GPU_offset[n], z + N3_GPU_offset[n] + D3)][k]);
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
			set_arrays_GPU(block[n][AMR_CHILD8], block[n][AMR_GPU]);
			GPU_write(block[n][AMR_CHILD8]);
			#endif
		}

	//Clean up memory of parent block
	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	GPU_boundprim(1);
	#endif

}

double B1_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double a0, a1, a2, a3, a11, a12, a13, b12, c13;
	double a23, a123, a113, a112, c123, b123;
	double B1p, B1m, d2B1p, d2B1m, d3B1p, d3B1m, d1B2p, d1B2m, d1B3p, d1B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	B1p = pb[n][index(n, i + D1, j, z)][1];
	B1m = pb[n][index(n, i, j, z)][1];

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[n][index(n, i + D1, j - D2, z)][1], pb[n][index(n, i + D1, j, z)][1], pb[n][index(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[n][index(n, i, j - D2, z)][1], pb[n][index(n, i, j, z)][1], pb[n][index(n, i, j + D2, z)][1]);
	
	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[n][index(n, i + D1, j, z - D3)][1], pb[n][index(n, i + D1, j, z)][1], pb[n][index(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[n][index(n, i, j, z - D3)][1], pb[n][index(n, i, j, z)][1], pb[n][index(n, i, j, z + D3)][1]);

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[n][index(n, i - D1, j + D2, z)][2], pb[n][index(n, i, j + D2, z)][2], pb[n][index(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[n][index(n, i - D1, j, z)][2], pb[n][index(n, i, j, z)][2], pb[n][index(n, i + D1, j, z)][2]);

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[n][index(n, i - D1, j, z + D3)][3], pb[n][index(n, i, j, z + D3)][3], pb[n][index(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[n][index(n, i - D1, j, z)][3], pb[n][index(n, i, j, z)][3], pb[n][index(n, i + D1, j, z)][3]);
	
	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	a113 = -b123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][1] * dx[n][3];
	a112 = -c123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][1] * dx[n][2];
	a1 = B1p - B1m;
	a2 = 0.5*(d2B1p + d2B1m) + c123 * dx[n][1] / (16. * dx[n][3]);
	a3 = 0.5*(d3B1p + d3B1m) + b123 * dx[n][1] / (16. * dx[n][2]);
	a23 = 0.5*(d23B1p + d23B1m);
	a12 = d2B1p - d2B1m;
	b12 = d1B2p - d1B2m;
	a13 = d3B1p - d3B1m;
	c13 = d1B3p - d1B3m;
	a11 = -0.5*(b12 / dx[n][2] + c13 / dx[n][3])*dx[n][1];
	a0 = 0.5*(B1p + B1m) - a11 / 4.;

	double var = a0 + a1*offset_1 + a2*offset_2 + a3*offset_3 + a11*offset_1*offset_1 + a12 * offset_1*offset_2 + a13*offset_1*offset_3 +
		a23*offset_2*offset_3 + a123*offset_1*offset_2*offset_3 + a113*offset_1*offset_1*offset_3 + a112*offset_1*offset_1*offset_2;
	return var;
}


double B2_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double b0, b1, b2, b3, b12, b13, b22, b23, a123, b123, c123, b223, b122;
	double B2p, B2m, a12, c23, d1B2p, d1B2m, d3B2p, d3B2m, d2B1p, d2B1m, d2B3p, d2B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[n][index(n, i - D1, j + D2, z)][2], pb[n][index(n, i, j + D2, z)][2], pb[n][index(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[n][index(n, i - D1, j, z)][2], pb[n][index(n, i, j, z)][2], pb[n][index(n, i + D1, j, z)][2]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[n][index(n, i, j + D2, z - D3)][2], pb[n][index(n, i, j + D2, z)][2], pb[n][index(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[n][index(n, i, j, z - D3)][2], pb[n][index(n, i, j, z)][2], pb[n][index(n, i, j, z + D3)][2]);

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[n][index(n, i + D1, j - D2, z)][1], pb[n][index(n, i + D1, j, z)][1], pb[n][index(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[n][index(n, i, j - D2, z)][1], pb[n][index(n, i, j, z)][1], pb[n][index(n, i, j + D2, z)][1]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[n][index(n, i, j - D2, z + D3)][3], pb[n][index(n, i, j, z + D3)][3], pb[n][index(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[n][index(n, i, j - D2, z)][3], pb[n][index(n, i, j, z)][3], pb[n][index(n, i, j + D2, z)][3]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	B2p = pb[n][index(n, i, j + D2, z)][2];
	B2m = pb[n][index(n, i, j, z)][2];

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	b223 = -a123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][2] * dx[n][2] * dx[n][3];
	b122 = -c123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][2] * dx[n][2];
	b1 = 0.5*(d1B2p + d1B2m) + c123 * dx[n][2] / (16. * dx[n][3]);
	b2 = B2p - B2m;
	b3 = 0.5*(d3B2p + d3B2m) + a123 * dx[n][2] / (16. * dx[n][1]);
	b13 = 0.5*(d13B2p + d13B2m);
	b12 = d1B2p - d1B2m;
	b23 = d3B2p - d3B2m;
	a12 = d2B1p - d2B1m;
	c23 = d2B3p - d2B3m;
	b22 = -0.5*(a12 / dx[n][1] + c23 / dx[n][3])*dx[n][2];
	b0 = 0.5*(B2p + B2m) - b22 / 4.;
	double var = b0 + b1*offset_1 + b2*offset_2 + b3*offset_3 + b12*offset_1*offset_2 + b22*offset_2*offset_2 + b23*offset_2*offset_3 +
		b13*offset_1*offset_3 + b223*offset_2*offset_2*offset_3 + b123*offset_1*offset_2*offset_3 + b122*offset_1*offset_2*offset_2;
	return var;
}

double B3_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double c0, c1, c2, c3, c12, c13, c23, c33, a13, b23, a123, b123,c123, c233, c133;
	double B3p, B3m, d1B3p, d1B3m, d2B3p, d2B3m, d3B1p, d3B1m, d3B2p, d3B2m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[n][index(n, i - D1, j, z + D3)][3], pb[n][index(n, i, j, z + D3)][3], pb[n][index(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[n][index(n, i - D1, j, z)][3], pb[n][index(n, i, j, z)][3], pb[n][index(n, i + D1, j, z)][3]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[n][index(n, i, j - D2, z + D3)][3], pb[n][index(n, i, j, z + D3)][3], pb[n][index(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[n][index(n, i, j - D2, z)][3], pb[n][index(n, i, j, z)][3], pb[n][index(n, i, j + D2, z)][3]);

	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[n][index(n, i + D1, j, z - D3)][1], pb[n][index(n, i + D1, j, z)][1], pb[n][index(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[n][index(n, i, j, z - D3)][1], pb[n][index(n, i, j, z)][1], pb[n][index(n, i, j, z + D3)][1]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[n][index(n, i, j + D2, z - D3)][2], pb[n][index(n, i, j + D2, z)][2], pb[n][index(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[n][index(n, i, j, z - D3)][2], pb[n][index(n, i, j, z)][2], pb[n][index(n, i, j, z + D3)][2]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	B3p = pb[n][index(n, i, j, z + D3)][3];
	B3m = pb[n][index(n, i, j, z)][3];

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	c233 = -a123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][2] * dx[n][3] * dx[n][3];;
	c133 = -b123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][3] * dx[n][3];;
	c1 = 0.5*(d1B3p + d1B3m) + b123 * dx[n][3] / (16.*dx[n][2]);
	c2 = 0.5*(d2B3p + d2B3m) + a123 * dx[n][3] / (16.*dx[n][1]);
	c3 = B3p - B3m;
	c12 = 0.5*(d12B3p + d12B3m);
	c13 = d1B3p - d1B3m;
	c23 = d2B3p - d2B3m;
	a13 = d3B1p - d3B1m;
	b23 = d3B2p - d3B2m;
	c33 = -0.5*(a13 / dx[n][1] + b23 / dx[n][2])*dx[n][3];
	c0 = 0.5*(B3p + B3m) - c33 / 4.;

	double var = c0 + c1*offset_1 + c2*offset_2 + c3*offset_3 + c13*offset_1*offset_3 + c23*offset_2*offset_3 + c33*offset_3*offset_3 +
		c12*offset_1*offset_2 + c233*offset_2*offset_3*offset_3 + c133*offset_1*offset_3*offset_3 + c123*offset_1*offset_2*offset_3;
	return var;
}

//Checks if neighbouring blocks are sufficiently refined so that no double jumps in refinement level are created
void check_nesting(int n){
	int i,z;
	for (i = AMR_NBR1; i <= AMR_CORN12; i++){
		if (block[n][i] >= 0 && block[block[n][i]][AMR_PARENT] >= 0 && block[block[block[n][i]][AMR_PARENT]][AMR_ACTIVE] == 1){
			refine(block[block[n][i]][AMR_PARENT]);
		}
	}

	//Make sure that the transmissive boundary works ok at the pole
	/*#if(TRANS_BOUND)
	if (block[n][AMR_POLE] > 0){
		for (z = 0; z < pow(1 + REF_3, block[n][AMR_LEVEL]); z++){
			if (block[][AMR_LEVEL] != block[n][AMR_LEVEL] + 1){
				refine(block[]);
			}
		}

	}
	#endif*/
}

#if WHICHPROBLEM==DISRUPTION_PROBLEM
#define DENSITY_CUTOFF 0.0000001
#else
#define DENSITY_CUTOFF 0.5
#endif

//Refine on basis of some criteria
void check_refcrit(void){
	int n, task, i, l, level, number;
	int node, n_send, n_blocks;
	double rhomax[NB], rho_rec;
	int NODE[NB];
	if (max_levels == 0) max_levels = N_LEVELS;
	int tag;
	int count;
	int begin1, end1;

	//First close dump files in progress
	int u;
	int u_stride = 200;
	int u_max = (n_active_total - n_active_total%u_stride) / u_stride;
	if (n_active_total%u_stride != 0) u_max++;

	if (first_dump == 1){
		for (n = 0; n < n_active; n++){
			MPI_Wait(&req_block[n_ord[n]][0], &Statbound[n_ord[n]][0]);
			if (dump_cnt % 1 == 0){
				MPI_Wait(&req_blockdiag[n_ord[n]][0], &Statbound[n_ord[n]][1]);
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
			MPI_Wait(&req_block_rdump[n_ord[n]][0], &Statbound[n_ord[n]][0]);
			MPI_File_close(&rdump[n_ord[n]]);
		}
	}
	first_rdump = 0;

	MPI_Barrier(mpi_cartcomm);
	begin1 = time(NULL);
	do{
		tag = 0;
		count = 0;
		pre_refine();
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
						rc = MPI_Isend(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, task, n_ord_total[n] % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
						MPI_Request_free(&req[0]);
					}
				}
			}
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				rc = MPI_Irecv(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], n_ord_total[n] % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
			}
		}
		for (n = 0; n < n_active_total; n++){
			if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
				MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[n_ord[0]][0]);
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
						refine(n_ord_total[n]);
						count++;
						NODE[block[n_ord_total[n]][AMR_NODE]] += 7;
					}
				}
			}
		}
		//MPI_Barrier(mpi_cartcomm);

		post_refine();
		if (tag != 0 && n_active_total<numtasks*MAX_BLOCKS && count>0){
			balance_load();
			#if(GPU_ENABLED)
			balance_load_gpu();
			#endif
		}
	} while (tag != 0 && n_active_total<numtasks*MAX_BLOCKS && count>0);

	if (tag==1 && count==0) fprintf(stderr, "Maximum number of blocks exceeded, refinement capped so refinement criterion can not anymore be honoured by H-AMR. Please select more nodes or adjust refinement criterion! \n");

	pre_refine();
	count = 0;
	
	//First make sure all nodes have the same rhomax
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
			rhomax[n_ord_total[n]] = calc_rhomax(n_ord_total[n]);
			for (task = 0; task<numtasks; task++){
				if (rank != task){
					rc = MPI_Isend(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, task, n_ord_total[n] % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
					MPI_Request_free(&req[0]);
				}
			}
		}
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			rc = MPI_Irecv(&rhomax[n_ord_total[n]], 1, MPI_DOUBLE, block[n_ord_total[n]][AMR_NODE], n_ord_total[n]%MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
		}
	}
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[n_ord[0]][0]);
		}
	}

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
						rc += MPI_Isend(&p[n_send][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (50 * NB + n_send) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_send][598]);
						#if STAGGERED
						rc += MPI_Isend(&ps[n_send][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, node, (51 * NB + n_send) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_send][597]);
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
						set_arrays(n_send);
						set_grid(n_send);
						if (block[n_send][AMR_NODE] >= 0){
							rc += MPI_Irecv(&p[n_send][0], NPR*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (50 * NB + n_send) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_send][596]);
							#if STAGGERED
							rc += MPI_Irecv(&ps[n_send][0], NDIM*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G), MPI_DOUBLE, block[n_send][AMR_NODE], (51 * NB + n_send) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[n_send][595]);
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
						MPI_Wait(&boundreqs[n_send][598], &Statbound[n_send][0]);
						#if STAGGERED
						MPI_Wait(&boundreqs[n_send][597], &Statbound[n_send][1]);
						#endif
						#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
							MPI_Wait(&boundreqs[n_send][596], &Statbound[n_send][10]);
							#if STAGGERED
							MPI_Wait(&boundreqs[n_send][595], &Statbound[n_send][11]);
							#endif
						}
						#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
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
			derefine(block[n_ord_total[n]][AMR_PARENT]);
			count++;
			block[block[n_ord_total[n]][AMR_PARENT]][AMR_TAG] = 0;
			//block[block[n_ord_total[n]][AMR_PARENT]][AMR_TIMELEVEL] = block[block[block[n_ord_total[n]][AMR_PARENT]][AMR_CHILD1]][AMR_TIMELEVEL];
		}
	}

	post_refine();

	balance_load();
	#if(GPU_ENABLED)
	balance_load_gpu();
	#endif

	MPI_Barrier(mpi_cartcomm);
	end1 = time(NULL);
	if (rank == 0) fprintf(stderr, "Runtime load balance: %f \n", (double)(end1 - begin1));

	//Start very conservatively
	dt /= 2.;
	for (n = 0; n < n_active_total; n ++){
		block[n_ord_total[n]][AMR_TIMELEVEL] = 1;
	}
}

double calc_rhomax(int n){
	int i, j, z;
	double rhomax=0.0;
	if (block[n][AMR_NODE] == rank){
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) {
			if (p[n][index(n, i, j, z)][RHO] > rhomax) rhomax = p[n][index(n, i, j, z)][RHO];
		}
	}
	return rhomax;
}

//Calculates RAM requirements in bytes (conservatively)
double calc_mem(int n_blocks){
	return n_blocks*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * 1000;
}