#include "include.h"
#include "decs_MPI.h"

//Wrapper for calculation of GPU memory
#if(GPU_DIRECT)
#define gpuMem(val1,val2, val3) (double)(val2*(val3==1))
#else
#define gpuMem(val1,val2, val3) (double)(0.0)
#endif

/*****************************************************************/
/*****************************************************************
  set_arrays():
  ----------

       -- sets to zero all arrays, plus performs pointer trick
          so that grid arrays can legitimately refer to ghost
          zone quantities at positions  i = -2, -1, N1, N1+1 and
          j = -2, -1, N2, N2+1

 *****************************************************************/
void set_arrays_image(void)
{

}

void set_arrays(int n)
{
	int i=0;

	//Find location in memory for new block and set nl[n]
	while (i < NB_LOCAL){
		if (mem_spot[i] != 1) break;
		i++;
		if (i == NB_LOCAL){
			fprintf(stderr, "Node %d ran out of local node memory! Stopping \n", rank);
			exit(0);
		}
	}
	nl[n] = i;

	if (mem_spot[i] == 0){
		alloc_bounds_CPU(n);
		mem_spot[i] = 1;
		return;
	}
	else mem_spot[i] = 1;

	array[nl[n]] = (float *)malloc(NPRDUMP * BS_1*BS_2*BS_3 * sizeof(float));
	#if(DUMP_SMALL)
	array_reduced[nl[n]] = (float *)malloc(NPRDUMP * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 * sizeof(float));
	array_gdump1_reduced[nl[n]] = (double *)malloc(9 * BS_1 / REDUCE_FACTOR1 *BS_2 / REDUCE_FACTOR2 *BS_3 / REDUCE_FACTOR3 * sizeof(double));
	array_gdump2_reduced[nl[n]] = (double *)malloc(49 * BS_1 / REDUCE_FACTOR1 *BS_2 / REDUCE_FACTOR2 * (!NSY + NSY * BS_3 / REDUCE_FACTOR3) * sizeof(double));
	#endif
	array_gdump1[nl[n]] = (double *)malloc(9 * BS_1*BS_2*BS_3 * sizeof(double));
	array_gdump2[nl[n]] = (double *)malloc(49 * BS_1 * BS_2 * (!NSY + NSY * BS_3) * sizeof(double));
	array_rdump[nl[n]] = (double *)malloc((NPR + NDIM) * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	array_diag[nl[n]] = (float *)malloc(4 * BS_1*BS_2*BS_3 * sizeof(float));
	Katm[nl[n]] = (double(*))malloc((BS_1 + 2 * N1G) * sizeof(double));
	p[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	ph[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	U[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#if(DO_IMEX && RAD_M1)
	U_n[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	U_0[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	U_1[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	dU_RAD0[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	dU_RAD1[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#else
	U[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#endif
	#if(STAGGERED)
	ps[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	psh[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	#if(LEER)
	V[nl[n]] = (double(*)[6])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double[6]));
	#endif
	dq[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F1[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F2[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	F3[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	pflag[nl[n]] = (int(*))malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int));
	#if(RAD_M1)
	pflag_rad[nl[n]] = (int(*))malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(int));
	#endif
	#if(CARTESIAN_GR)
	pflag_cart[nl[n]] = (int(*))malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(int));
	#endif
	#if(DO_RBOUND || (NEUTRON_STAR && 0))
	pflag_rbound[nl[n]] = (int(*))malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(int));
	#endif
	#if(NEUTRON_STAR && !USE_PS1START)
	Bx1_surface[nl[n]] = (double(*))malloc((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double));
	#endif
	#if(CPU_OPENMP || 1)
	#if(STAGGERED)
	dE[nl[n]] = (double(*)[2][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[2][NDIM][NDIM]));
	#endif
	E_corn[nl[n]] = (double(*)[NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	failimage[nl[n]] = (int(*)[NFAIL])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int[NFAIL]));
	#if (NEUTRINOS_DEBUG)
	allflags_NU[nl[n]] = (double(*)[NU_SPECIES])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NU_SPECIES]));
	#endif
	#if(!NSY)
	conn[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NDIM][NDIM][NDIM]));
	gcov[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	gcon[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	gdet[nl[n]] = (double(*)[NPG])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG]));
	#else
	conn[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	gcov[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NPG][NDIM][NDIM]));
	gcon[nl[n]] = (double(*)[NPG][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG][NDIM][NDIM]));
	gdet[nl[n]] = (double(*)[NPG])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG]));
	#endif
	#if(FRAME_TRANSFORM)
	Mud[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	Mud_inv[nl[n]] = (double(*)[NDIM][NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	#endif
	#if(ZIRI_DUMP)
	dump_buffer[nl[n]] = (double(*))malloc(BS_1 * BS_2 * BS_3 * 13 *sizeof(double));
	dxdxp_z[nl[n]] = (double(*)[NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	dxpdx_z[nl[n]] = (double(*)[NDIM][NDIM])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	#endif
	#if (ELLIPTICAL2)
	dU_s[nl[n]] = (double(*)[NPR])malloc((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPR]));
	#endif
	alloc_bounds_CPU(n);
}

void alloc_bounds_CPU(int n){
	int ref1_1, ref1_3, ref1_5, ref1_6;
	int ref2_2, ref2_4, ref2_5, ref2_6;
	int ref3_1, ref3_2, ref3_3, ref3_4;
	int ref1_1s, ref1_3s;
	int ref3_1s, ref3_3s, ref3_2s, ref3_4s;

	ref1_1 = REF_1; ref1_3 = REF_1; ref1_5 = REF_1; ref1_6 = REF_1;
	ref2_2 = REF_2; ref2_4 = REF_2; ref2_5 = REF_2; ref2_6 = REF_2;
	ref3_1 = REF_3; ref3_2 = REF_3; ref3_3 = REF_3; ref3_4 = REF_3;
	ref1_1s = REF_1; ref1_3s = REF_1;
	ref3_1s = REF_3; ref3_3s = REF_3;
	ref3_2s = REF_3; ref3_4s = REF_3;

	if (block[n][AMR_LEVEL] != N_LEVELS - 1){
		if (block[n][AMR_NBR1_3] >= 0) ref1_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR3_1] >= 0) ref1_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR1_3] >= 0) ref3_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_NBR3_1] >= 0) ref3_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
	}
	ref1_1s = ref1_1;
	ref1_3s = ref1_3;
	ref3_1s = ref3_1;
	ref3_3s = ref3_3;

	if (block[n][AMR_NBR1P] >= 0)ref1_1s = MY_MIN(ref1_1, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR1P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR3P] >= 0)ref1_3s = MY_MIN(ref1_3, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR3P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR1P] >= 0)ref3_1s = MY_MIN(ref3_1, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR1P]][AMR_LEVEL3]);
	if (block[n][AMR_NBR3P] >= 0)ref3_3s = MY_MIN(ref3_3, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR3P]][AMR_LEVEL3]);
	if ((block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*(int)pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) && DEREFINE_POLE){
		ref3_2s = 0;
		ref3_4s = 0;
	}

	send1[nl[n]] = (double *)malloc(NG *(1 + ref1_1)*(1 + ref3_1)* (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	send2[nl[n]] = (double *)malloc(NG *(1 + ref2_2)*(1 + ref3_2)* (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	send3[nl[n]] = (double *)malloc(NG *(1 + ref1_3)*(1 + ref3_3)* (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	send4[nl[n]] = (double *)malloc(NG *(1 + ref2_4)*(1 + ref3_4)* (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5[nl[n]] = (double *)malloc(NG *(1 + ref2_5)*(1 + ref1_5)* (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G) * (BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	send6[nl[n]] = (double *)malloc(NG *(1 + ref2_6)*(1 + ref1_6)* (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G) * (BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	#endif

	#if(N_LEVELS>1)
	send1_3[nl[n]] = send1[nl[n]];
	send1_4[nl[n]] = send1[nl[n]] + ref3_1*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send1_7[nl[n]] = send1[nl[n]] + (ref3_1 + ref1_1)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send1_8[nl[n]] = send1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	send2_1[nl[n]] = send2[nl[n]];
	send2_2[nl[n]] = send2[nl[n]] + ref3_2*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send2_3[nl[n]] = send2[nl[n]] + (ref3_2 + ref2_2)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send2_4[nl[n]] = send2[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	send3_1[nl[n]] = send3[nl[n]];
	send3_2[nl[n]] = send3[nl[n]] + ref3_3*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send3_5[nl[n]] = send3[nl[n]] + (ref3_3 + ref1_3)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send3_6[nl[n]] = send3[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	send4_5[nl[n]] = send4[nl[n]];
	send4_6[nl[n]] = send4[nl[n]] + ref3_4*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	send4_7[nl[n]] = send4[nl[n]] + (ref3_4 + ref2_4)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	send4_8[nl[n]] = send4[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	#if(N3G>0)
	send5_1[nl[n]] = send5[nl[n]];
	send5_3[nl[n]] = send5[nl[n]] + ref2_5*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send5_5[nl[n]] = send5[nl[n]] + (ref1_5 + ref2_5)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send5_7[nl[n]] = send5[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	send6_2[nl[n]] = send6[nl[n]];
	send6_4[nl[n]] = send6[nl[n]] + ref2_6*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	send6_6[nl[n]] = send6[nl[n]] + (ref1_6 + ref2_6)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	send6_8[nl[n]] = send6[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	#endif
	#endif
	receive1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	receive6[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_4[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_7[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive1_8[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	receive2_1[nl[n]] = receive2[nl[n]];
	receive2_2[nl[n]] = receive2[nl[n]] + ref3_4*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_3[nl[n]] = receive2[nl[n]] + (ref3_4 + ref2_4)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_4[nl[n]] = receive2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive3_1[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_2[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_5[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive3_6[nl[n]] = (double *)malloc(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	receive4_5[nl[n]] = receive4[nl[n]];
	receive4_6[nl[n]] = receive4[nl[n]] + ref3_2*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_7[nl[n]] = receive4[nl[n]] + (ref3_2 + ref2_2)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_8[nl[n]] = receive4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)
	receive5_1[nl[n]] = receive5[nl[n]];
	receive5_3[nl[n]] = receive5[nl[n]] + ref2_6*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_5[nl[n]] = receive5[nl[n]] + (ref1_6 + ref2_6)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_7[nl[n]] = receive5[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive6_2[nl[n]] = receive6[nl[n]];
	receive6_4[nl[n]] = receive6[nl[n]] + ref2_5*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_6[nl[n]] = receive6[nl[n]] + (ref1_5 + ref2_5)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_8[nl[n]] = receive6[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#if(PRESTEP==-100 || PRESTEP2==-100)
	tempreceive1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* 2* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	tempreceive2[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	tempreceive3[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)*2* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	tempreceive4[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	tempreceive5[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	tempreceive6[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	tempreceive1_3[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_4[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_7[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive1_8[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	tempreceive2_1[nl[n]] = tempreceive2[nl[n]];
	tempreceive2_2[nl[n]] = tempreceive2[nl[n]] + ref3_4*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive2_3[nl[n]] = tempreceive2[nl[n]] + (ref3_4 + ref2_4)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive2_4[nl[n]] = tempreceive2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempreceive3_1[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_2[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_5[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive3_6[nl[n]] = (double *)malloc(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	tempreceive4_5[nl[n]] = tempreceive4[nl[n]];
	tempreceive4_6[nl[n]] = tempreceive4[nl[n]] + ref3_2*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempreceive4_7[nl[n]] = tempreceive4[nl[n]] + (ref3_2 + ref2_2)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempreceive4_8[nl[n]] = tempreceive4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)
	tempreceive5_1[nl[n]] = tempreceive5[nl[n]];
	tempreceive5_3[nl[n]] = tempreceive5[nl[n]] + ref2_6*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive5_5[nl[n]] = tempreceive5[nl[n]] + (ref1_6 + ref2_6)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive5_7[nl[n]] = tempreceive5[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	tempreceive6_2[nl[n]] = tempreceive6[nl[n]];
	tempreceive6_4[nl[n]] = tempreceive6[nl[n]] + ref2_5*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	tempreceive6_6[nl[n]] = tempreceive6[nl[n]] + (ref1_5 + ref2_5)*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	tempreceive6_8[nl[n]] = tempreceive6[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(2*NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#endif
	send1_fine[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_fine[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_fine[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_fine[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3)* (NPR)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_fine[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4)* (NPR)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_fine[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1)* (NPR)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_fine[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2)* (NPR)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_fine[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6)* (NPR)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	receive6_fine[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5)* (NPR)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3fine[nl[n]] = receive1_fine[nl[n]];
	receive1_4fine[nl[n]] = receive1_fine[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive1_7fine[nl[n]] = receive1_fine[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive1_8fine[nl[n]] = receive1_fine[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	receive2_1fine[nl[n]] = receive2_fine[nl[n]];
	receive2_2fine[nl[n]] = receive2_fine[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_3fine[nl[n]] = receive2_fine[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive2_4fine[nl[n]] = receive2_fine[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	receive3_1fine[nl[n]] = receive3_fine[nl[n]];
	receive3_2fine[nl[n]] = receive3_fine[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive3_5fine[nl[n]] = receive3_fine[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive3_6fine[nl[n]] = receive3_fine[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	receive4_5fine[nl[n]] = receive4_fine[nl[n]];
	receive4_6fine[nl[n]] = receive4_fine[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_7fine[nl[n]] = receive4_fine[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	receive4_8fine[nl[n]] = receive4_fine[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)
	receive5_1fine[nl[n]] = receive5_fine[nl[n]];
	receive5_3fine[nl[n]] = receive5_fine[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_5fine[nl[n]] = receive5_fine[nl[n]] + (ref1_6 + ref2_6)*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive5_7fine[nl[n]] = receive5_fine[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(NPR *(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G));
	receive6_2fine[nl[n]] = receive6_fine[nl[n]];
	receive6_4fine[nl[n]] = receive6_fine[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_6fine[nl[n]] = receive6_fine[nl[n]] + (ref1_5 + ref2_5)*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	receive6_8fine[nl[n]] = receive6_fine[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(NPR *(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G));
	#endif
	#endif
	#if(CPU_OPENMP)
	send1_flux[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_flux[nl[n]] = (double *)malloc(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_flux[nl[n]] = (double *)malloc(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_flux[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive2_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	receive3_flux[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive4_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	receive5_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	receive6_flux[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3flux[nl[n]] = receive1_flux[nl[n]];
	receive1_4flux[nl[n]] = receive1_flux[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_7flux[nl[n]] = receive1_flux[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_8flux[nl[n]] = receive1_flux[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive2_1flux[nl[n]] = receive2_flux[nl[n]];
	receive2_2flux[nl[n]] = receive2_flux[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_3flux[nl[n]] = receive2_flux[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_4flux[nl[n]] = receive2_flux[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive3_1flux[nl[n]] = receive3_flux[nl[n]];
	receive3_2flux[nl[n]] = receive3_flux[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_5flux[nl[n]] = receive3_flux[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_6flux[nl[n]] = receive3_flux[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive4_5flux[nl[n]] = receive4_flux[nl[n]];
	receive4_6flux[nl[n]] = receive4_flux[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_7flux[nl[n]] = receive4_flux[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_8flux[nl[n]] = receive4_flux[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	receive5_1flux[nl[n]] = receive5_flux[nl[n]];
	receive5_3flux[nl[n]] = receive5_flux[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_5flux[nl[n]] = receive5_flux[nl[n]] + (ref2_6 + ref1_6)*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_7flux[nl[n]] = receive5_flux[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive6_2flux[nl[n]] = receive6_flux[nl[n]];
	receive6_4flux[nl[n]] = receive6_flux[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_6flux[nl[n]] = receive6_flux[nl[n]] + (ref2_5 + ref1_5)*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_8flux[nl[n]] = receive6_flux[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	#endif
	receive1_flux1[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive2_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	receive3_flux1[nl[n]] = (double *)malloc(NPR* (BS_1)*(BS_3) * sizeof(double));
	receive4_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	receive5_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	receive6_flux1[nl[n]] = (double *)malloc(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3flux1[nl[n]] = receive1_flux1[nl[n]];
	receive1_4flux1[nl[n]] = receive1_flux1[nl[n]] + ref3_3*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_7flux1[nl[n]] = receive1_flux1[nl[n]] + (ref3_3 + ref1_3)*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive1_8flux1[nl[n]] = receive1_flux1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR *(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	receive2_1flux1[nl[n]] = receive2_flux1[nl[n]];
	receive2_2flux1[nl[n]] = receive2_flux1[nl[n]] + ref3_4*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_3flux1[nl[n]] = receive2_flux1[nl[n]] + (ref3_4 + ref2_4)*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive2_4flux1[nl[n]] = receive2_flux1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR *(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	receive3_1flux1[nl[n]] = receive3_flux1[nl[n]];
	receive3_2flux1[nl[n]] = receive3_flux1[nl[n]] + ref3_1*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_5flux1[nl[n]] = receive3_flux1[nl[n]] + (ref3_1 + ref1_1)*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive3_6flux1[nl[n]] = receive3_flux1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR *(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	receive4_5flux1[nl[n]] = receive4_flux1[nl[n]];
	receive4_6flux1[nl[n]] = receive4_flux1[nl[n]] + ref3_2*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_7flux1[nl[n]] = receive4_flux1[nl[n]] + (ref3_2 + ref2_2)*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	receive4_8flux1[nl[n]] = receive4_flux1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR *(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	receive5_1flux1[nl[n]] = receive5_flux1[nl[n]];
	receive5_3flux1[nl[n]] = receive5_flux1[nl[n]] + ref2_6*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_5flux1[nl[n]] = receive5_flux1[nl[n]] + (ref2_6 + ref1_6)*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive5_7flux1[nl[n]] = receive5_flux1[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR *(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	receive6_2flux1[nl[n]] = receive6_flux1[nl[n]];
	receive6_4flux1[nl[n]] = receive6_flux1[nl[n]] + ref2_5*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_6flux1[nl[n]] = receive6_flux1[nl[n]] + (ref2_5 + ref1_5)*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	receive6_8flux1[nl[n]] = receive6_flux1[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR *(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	receive1_3flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_4flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_7flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive1_8flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	receive2_1flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_2flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_3flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive2_4flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	receive3_1flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_2flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_5flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive3_6flux2[nl[n]] = (double *)malloc(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	receive4_5flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_6flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_7flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	receive4_8flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	#if(N3G>0)
	receive5_1flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_3flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_5flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive5_7flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	receive6_2flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_4flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_6flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	receive6_8flux2[nl[n]] = (double *)malloc(NPR*(BS_2 / (1 + ref2_5) + 2 * N2G) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	#endif
	#endif
	#endif
	#if(CPU_OPENMP || 1)
	send1_E[nl[n]] = (double *)malloc(2 * (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	send2_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	send3_E[nl[n]] = (double *)malloc(2 * (BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	send4_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	send5_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	send6_E[nl[n]] = (double *)malloc(2 * (BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	receive1_E[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_E[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_E[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_E[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_E[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_E[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3E[nl[n]] = receive1_E[nl[n]];
	receive1_4E[nl[n]] = receive1_E[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_7E[nl[n]] = receive1_E[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_8E[nl[n]] = receive1_E[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive2_1E[nl[n]] = receive2_E[nl[n]];
	receive2_2E[nl[n]] = receive2_E[nl[n]] + ref3_4*(2 * ((BS_2 + 2 * D2) / (1 + ref2_4))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_3E[nl[n]] = receive2_E[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_4E[nl[n]] = receive2_E[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive3_1E[nl[n]] = receive3_E[nl[n]];
	receive3_2E[nl[n]] = receive3_E[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_5E[nl[n]] = receive3_E[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_6E[nl[n]] = receive3_E[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive4_5E[nl[n]] = receive4_E[nl[n]];
	receive4_6E[nl[n]] = receive4_E[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_7E[nl[n]] = receive4_E[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_8E[nl[n]] = receive4_E[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	receive5_1E[nl[n]] = receive5_E[nl[n]];
	receive5_3E[nl[n]] = receive5_E[nl[n]] + ref2_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_5E[nl[n]] = receive5_E[nl[n]] + (ref2_6 + ref1_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_7E[nl[n]] = receive5_E[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive6_2E[nl[n]] = receive6_E[nl[n]];
	receive6_4E[nl[n]] = receive6_E[nl[n]] + ref2_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_6E[nl[n]] = receive6_E[nl[n]] + (ref2_5 + ref1_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_8E[nl[n]] = receive6_E[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	#endif
	receive1_E1[nl[n]] = (double *)malloc((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	receive2_E1[nl[n]] = (double *)malloc((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	receive3_E1[nl[n]] = (double *)malloc((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	receive4_E1[nl[n]] = (double *)malloc((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	receive5_E1[nl[n]] = (double *)malloc((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_E1[nl[n]] = (double *)malloc((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3E1[nl[n]] = receive1_E1[nl[n]];
	receive1_4E1[nl[n]] = receive1_E1[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_7E1[nl[n]] = receive1_E1[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive1_8E1[nl[n]] = receive1_E1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	receive2_1E1[nl[n]] = receive2_E1[nl[n]];
	receive2_2E1[nl[n]] = receive2_E1[nl[n]] + ref3_4*(2 * ((BS_2 + 2 * D2) / (1 + ref2_4))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_3E1[nl[n]] = receive2_E1[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive2_4E1[nl[n]] = receive2_E1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	receive3_1E1[nl[n]] = receive3_E1[nl[n]];
	receive3_2E1[nl[n]] = receive3_E1[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_5E1[nl[n]] = receive3_E1[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive3_6E1[nl[n]] = receive3_E1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	receive4_5E1[nl[n]] = receive4_E1[nl[n]];
	receive4_6E1[nl[n]] = receive4_E1[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_7E1[nl[n]] = receive4_E1[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	receive4_8E1[nl[n]] = receive4_E1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	receive5_1E1[nl[n]] = receive5_E1[nl[n]];
	receive5_3E1[nl[n]] = receive5_E1[nl[n]] + ref2_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_5E1[nl[n]] = receive5_E1[nl[n]] + (ref2_6 + ref1_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive5_7E1[nl[n]] = receive5_E1[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	receive6_2E1[nl[n]] = receive6_E1[nl[n]];
	receive6_4E1[nl[n]] = receive6_E1[nl[n]] + ref2_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_6E1[nl[n]] = receive6_E1[nl[n]] + (ref2_5 + ref1_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	receive6_8E1[nl[n]] = receive6_E1[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	receive1_3E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_4E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_7E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive1_8E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	receive2_1E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_2E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_3E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive2_4E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	receive3_1E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_2E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_5E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive3_6E2[nl[n]] = (double *)malloc(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	receive4_5E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_6E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_7E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	receive4_8E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive5_1E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_3E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_5E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive5_7E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	receive6_2E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_4E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_6E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	receive6_8E2[nl[n]] = (double *)malloc(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#endif
	send_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	send_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	send_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	send_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	send_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive_E3_corn1_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4_1[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8_1[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12_1[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	tempreceive_E3_corn1_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn2_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn3_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	tempreceive_E3_corn4_2[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn6_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn7_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E2_corn8_2[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	tempreceive_E1_corn9_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn10_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn11_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	tempreceive_E1_corn12_2[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_12[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_12[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_12[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	receive_E3_corn1_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn2_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn3_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	receive_E3_corn4_22[nl[n]] = (double *)malloc((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn6_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn7_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E2_corn8_22[nl[n]] = (double *)malloc((BS_2 + 2 * D2) * sizeof(double));
	receive_E1_corn9_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn10_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn11_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	receive_E1_corn12_22[nl[n]] = (double *)malloc((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#endif
	#endif
}

void free_arrays(int n){
	int i, count_node = 0, count_gpu = 0;

	//Count on node/GPU
	for (i = 0; i < NB_LOCAL; i++){
		if (mem_spot[i] != -1){
			count_node++; //Number of allocated blocks on node
			if (mem_spot_gpu[i] == block[n][AMR_GPU] && GPU_ENABLED==1) count_gpu++;
		}
	}
	#if(MEM_CLEAN)
	mem_spot[nl[n]] = -1;
	#else
	if (count_gpu < 0.8*(max_blocks/numtasks) || count_node < 0.8*(max_blocks/numtasks)){
		mem_spot[nl[n]] = 0;
		free_bound_cpu(n);
		return;
	}
	else mem_spot[nl[n]] = -1;
	#endif
	free(p[nl[n]]);
	free(ph[nl[n]]);
	#if(LEER)
	free(V[nl[n]]);
	#endif
	#if(STAGGERED)
	free(ps[nl[n]]);
	free(psh[nl[n]]);
	#endif
	#if(DO_IMEX && RAD_M1)
	free(U_n[nl[n]]);
	free(U_0[nl[n]]);
	free(U_1[nl[n]]);
	free(dU_MHD1[nl[n]]);
	free(dU_RAD0[nl[n]]);
	free(dU_RAD1[nl[n]]);
	#endif
	#if(RAD_M1)
	free(pflag_rad[nl[n]]);
	#endif
	#if(CARTESIAN_GR)
	free(pflag_cart[nl[n]]);
	#endif
	#if(DO_RBOUND || (NEUTRON_STAR && 0))
	free(pflag_rbound[nl[n]]);
	#endif
	#if(NEUTRON_STAR && !USE_PS1START)
	free(Bx1_surface[nl[n]]);
	#endif
	free(U[nl[n]]);
	free(dq[nl[n]]);
	free(F1[nl[n]]);
	free(F2[nl[n]]);
	free(F3[nl[n]]);
	free(pflag[nl[n]]);
	#if(GPU_DEBUG || CPU_OPENMP || 1)
	#if(STAGGERED)
	free(dE[nl[n]]);
	#endif
	free(E_corn[nl[n]]);
	#endif
	free(failimage[nl[n]]);
	#if (NEUTRINOS_DEBUG)
	free(allflags_NU[nl[n]]);
	#endif
	free(conn[nl[n]]);
	free(gcov[nl[n]]);
	free(gcon[nl[n]]);
	free(gdet[nl[n]]);
	#if(FRAME_TRANSFORM)
	free(Mud[nl[n]]);
	free(Mud_inv[nl[n]]);
	#endif
	#if(ZIRI_DUMP)
	free(dump_buffer[nl[n]]);
	free(dxdxp_z[nl[n]]);
	free(dxpdx_z[nl[n]]);
	#endif
	#if (ELLIPTICAL2)
	free(dU_s[nl[n]]);
	#endif
	free(Katm[nl[n]]);
	free(array[nl[n]]);
	#if(DUMP_SMALL)
	free(array_reduced[nl[n]]);
	free(array_gdump1_reduced[nl[n]]);
	free(array_gdump2_reduced[nl[n]]);
	#endif
	free(array_rdump[nl[n]]);
	free(array_gdump1[nl[n]]);
	free(array_gdump2[nl[n]]);
	free(array_diag[nl[n]]);
	free_bound_cpu(n);
}

void free_bound_cpu(int n){
	free(send1[nl[n]]);
	free(send2[nl[n]]);
	free(send3[nl[n]]);
	free(send4[nl[n]]);
	#if(N3G>0)
	free(send5[nl[n]]);
	free(send6[nl[n]]);
	#endif
	free(receive1[nl[n]]);
	#if(N_LEVELS>1)
	free(receive1_3[nl[n]]);
	free(receive1_4[nl[n]]);
	free(receive1_7[nl[n]]);
	free(receive1_8[nl[n]]);
	#endif
	free(receive2[nl[n]]);
	free(receive3[nl[n]]);
	#if(N_LEVELS>1)
	free(receive3_1[nl[n]]);
	free(receive3_2[nl[n]]);
	free(receive3_5[nl[n]]);
	free(receive3_6[nl[n]]);
	#endif
	free(receive4[nl[n]]);
	#if(N3G>0)
	free(receive5[nl[n]]);
	free(receive6[nl[n]]);
	#endif
	#if(PRESTEP==-100 || PRESTEP2==-100)
	free(tempreceive1[nl[n]]);
	#if(N_LEVELS>1)
	free(tempreceive1_3[nl[n]]);
	free(tempreceive1_4[nl[n]]);
	free(tempreceive1_7[nl[n]]);
	free(tempreceive1_8[nl[n]]);
	#endif
	free(tempreceive2[nl[n]]);
	free(tempreceive3[nl[n]]);
	#if(N_LEVELS>1)
	free(tempreceive3_1[nl[n]]);
	free(tempreceive3_2[nl[n]]);
	free(tempreceive3_5[nl[n]]);
	free(tempreceive3_6[nl[n]]);
	#endif
	free(tempreceive4[nl[n]]);
	#if(N3G>0)
	free(tempreceive5[nl[n]]);
	free(tempreceive6[nl[n]]);
	#endif

	#endif
	free(send1_fine[nl[n]]);
	free(send2_fine[nl[n]]);
	free(send3_fine[nl[n]]);
	free(send4_fine[nl[n]]);
	#if(N3G>0)
	free(send5_fine[nl[n]]);
	free(send6_fine[nl[n]]);
	#endif
	free(receive1_fine[nl[n]]);
	free(receive2_fine[nl[n]]);
	free(receive3_fine[nl[n]]);
	free(receive4_fine[nl[n]]);
	#if(N3G>0)
	free(receive5_fine[nl[n]]);
	free(receive6_fine[nl[n]]);
	#endif
	#if(CPU_OPENMP)
	free(send1_flux[nl[n]]);
	free(send2_flux[nl[n]]);
	free(send3_flux[nl[n]]);
	free(send4_flux[nl[n]]);
	#if(N3G>0)
	free(send5_flux[nl[n]]);
	free(send6_flux[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive1_flux[nl[n]]);
	free(receive2_flux[nl[n]]);
	free(receive3_flux[nl[n]]);
	free(receive4_flux[nl[n]]);
	free(receive5_flux[nl[n]]);
	free(receive6_flux[nl[n]]);
	free(receive1_flux1[nl[n]]);
	free(receive2_flux1[nl[n]]);
	free(receive3_flux1[nl[n]]);
	free(receive4_flux1[nl[n]]);
	free(receive5_flux1[nl[n]]);
	free(receive6_flux1[nl[n]]);
	free(receive1_3flux2[nl[n]]);
	free(receive1_4flux2[nl[n]]);
	free(receive1_7flux2[nl[n]]);
	free(receive1_8flux2[nl[n]]);
	free(receive2_1flux2[nl[n]]);
	free(receive2_2flux2[nl[n]]);
	free(receive2_3flux2[nl[n]]);
	free(receive2_4flux2[nl[n]]);
	free(receive3_1flux2[nl[n]]);
	free(receive3_2flux2[nl[n]]);
	free(receive3_5flux2[nl[n]]);
	free(receive3_6flux2[nl[n]]);
	free(receive4_5flux2[nl[n]]);
	free(receive4_6flux2[nl[n]]);
	free(receive4_7flux2[nl[n]]);
	free(receive4_8flux2[nl[n]]);
	#if(N3G>0)
	free(receive5_1flux2[nl[n]]);
	free(receive5_3flux2[nl[n]]);
	free(receive5_5flux2[nl[n]]);
	free(receive5_7flux2[nl[n]]);
	free(receive6_2flux2[nl[n]]);
	free(receive6_4flux2[nl[n]]);
	free(receive6_6flux2[nl[n]]);
	free(receive6_8flux2[nl[n]]);
	#endif
	#endif
	#endif
	#if(CPU_OPENMP || 1)
	free(send1_E[nl[n]]);
	free(send2_E[nl[n]]);
	free(send3_E[nl[n]]);
	free(send4_E[nl[n]]);
	#if(N3G>0)
	free(send5_E[nl[n]]);
	free(send6_E[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive1_E[nl[n]]);
	free(receive2_E[nl[n]]);
	free(receive3_E[nl[n]]);
	free(receive4_E[nl[n]]);
	free(receive5_E[nl[n]]);
	free(receive6_E[nl[n]]);
	free(receive1_E1[nl[n]]);
	free(receive2_E1[nl[n]]);
	free(receive3_E1[nl[n]]);
	free(receive4_E1[nl[n]]);
	free(receive5_E1[nl[n]]);
	free(receive6_E1[nl[n]]);
	free(receive1_3E2[nl[n]]);
	free(receive1_4E2[nl[n]]);
	free(receive1_7E2[nl[n]]);
	free(receive1_8E2[nl[n]]);
	free(receive2_1E2[nl[n]]);
	free(receive2_2E2[nl[n]]);
	free(receive2_3E2[nl[n]]);
	free(receive2_4E2[nl[n]]);
	free(receive3_1E2[nl[n]]);
	free(receive3_2E2[nl[n]]);
	free(receive3_5E2[nl[n]]);
	free(receive3_6E2[nl[n]]);
	free(receive4_5E2[nl[n]]);
	free(receive4_6E2[nl[n]]);
	free(receive4_7E2[nl[n]]);
	free(receive4_8E2[nl[n]]);
	#if(N3G>0)
	free(receive5_1E2[nl[n]]);
	free(receive5_3E2[nl[n]]);
	free(receive5_5E2[nl[n]]);
	free(receive5_7E2[nl[n]]);
	free(receive6_2E2[nl[n]]);
	free(receive6_4E2[nl[n]]);
	free(receive6_6E2[nl[n]]);
	free(receive6_8E2[nl[n]]);
	#endif
	#endif
	free(send_E3_corn1[nl[n]]);
	free(send_E3_corn2[nl[n]]);
	free(send_E3_corn3[nl[n]]);
	free(send_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(send_E2_corn5[nl[n]]);
	free(send_E2_corn6[nl[n]]);
	free(send_E2_corn7[nl[n]]);
	free(send_E2_corn8[nl[n]]);
	free(send_E1_corn9[nl[n]]);
	free(send_E1_corn10[nl[n]]);
	free(send_E1_corn11[nl[n]]);
	free(send_E1_corn12[nl[n]]);
	#endif
	free(receive_E3_corn1[nl[n]]);
	free(receive_E3_corn2[nl[n]]);
	free(receive_E3_corn3[nl[n]]);
	free(receive_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5[nl[n]]);
	free(receive_E2_corn6[nl[n]]);
	free(receive_E2_corn7[nl[n]]);
	free(receive_E2_corn8[nl[n]]);
	free(receive_E1_corn9[nl[n]]);
	free(receive_E1_corn10[nl[n]]);
	free(receive_E1_corn11[nl[n]]);
	free(receive_E1_corn12[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(receive_E3_corn1_1[nl[n]]);
	free(receive_E3_corn2_1[nl[n]]);
	free(receive_E3_corn3_1[nl[n]]);
	free(receive_E3_corn4_1[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_1[nl[n]]);
	free(receive_E2_corn6_1[nl[n]]);
	free(receive_E2_corn7_1[nl[n]]);
	free(receive_E2_corn8_1[nl[n]]);
	free(receive_E1_corn9_1[nl[n]]);
	free(receive_E1_corn10_1[nl[n]]);
	free(receive_E1_corn11_1[nl[n]]);
	free(receive_E1_corn12_1[nl[n]]);
	#endif
	free(receive_E3_corn1_2[nl[n]]);
	free(receive_E3_corn2_2[nl[n]]);
	free(receive_E3_corn3_2[nl[n]]);
	free(receive_E3_corn4_2[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_2[nl[n]]);
	free(receive_E2_corn6_2[nl[n]]);
	free(receive_E2_corn7_2[nl[n]]);
	free(receive_E2_corn8_2[nl[n]]);
	free(receive_E1_corn9_2[nl[n]]);
	free(receive_E1_corn10_2[nl[n]]);
	free(receive_E1_corn11_2[nl[n]]);
	free(receive_E1_corn12_2[nl[n]]);
	#endif
	#endif
	free(tempreceive_E3_corn1[nl[n]]);
	free(tempreceive_E3_corn2[nl[n]]);
	free(tempreceive_E3_corn3[nl[n]]);
	free(tempreceive_E3_corn4[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5[nl[n]]);
	free(tempreceive_E2_corn6[nl[n]]);
	free(tempreceive_E2_corn7[nl[n]]);
	free(tempreceive_E2_corn8[nl[n]]);
	free(tempreceive_E1_corn9[nl[n]]);
	free(tempreceive_E1_corn10[nl[n]]);
	free(tempreceive_E1_corn11[nl[n]]);
	free(tempreceive_E1_corn12[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	free(tempreceive_E3_corn1_1[nl[n]]);
	free(tempreceive_E3_corn2_1[nl[n]]);
	free(tempreceive_E3_corn3_1[nl[n]]);
	free(tempreceive_E3_corn4_1[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_1[nl[n]]);
	free(tempreceive_E2_corn6_1[nl[n]]);
	free(tempreceive_E2_corn7_1[nl[n]]);
	free(tempreceive_E2_corn8_1[nl[n]]);
	free(tempreceive_E1_corn9_1[nl[n]]);
	free(tempreceive_E1_corn10_1[nl[n]]);
	free(tempreceive_E1_corn11_1[nl[n]]);
	free(tempreceive_E1_corn12_1[nl[n]]);
	#endif
	free(tempreceive_E3_corn1_2[nl[n]]);
	free(tempreceive_E3_corn2_2[nl[n]]);
	free(tempreceive_E3_corn3_2[nl[n]]);
	free(tempreceive_E3_corn4_2[nl[n]]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_2[nl[n]]);
	free(tempreceive_E2_corn6_2[nl[n]]);
	free(tempreceive_E2_corn7_2[nl[n]]);
	free(tempreceive_E2_corn8_2[nl[n]]);
	free(tempreceive_E1_corn9_2[nl[n]]);
	free(tempreceive_E1_corn10_2[nl[n]]);
	free(tempreceive_E1_corn11_2[nl[n]]);
	free(tempreceive_E1_corn12_2[nl[n]]);
	#endif
	free(receive_E3_corn1_12[nl[n]]);
	free(receive_E3_corn2_12[nl[n]]);
	free(receive_E3_corn3_12[nl[n]]);
	free(receive_E3_corn4_12[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_12[nl[n]]);
	free(receive_E2_corn6_12[nl[n]]);
	free(receive_E2_corn7_12[nl[n]]);
	free(receive_E2_corn8_12[nl[n]]);
	free(receive_E1_corn9_12[nl[n]]);
	free(receive_E1_corn10_12[nl[n]]);
	free(receive_E1_corn11_12[nl[n]]);
	free(receive_E1_corn12_12[nl[n]]);
	#endif
	free(receive_E3_corn1_22[nl[n]]);
	free(receive_E3_corn2_22[nl[n]]);
	free(receive_E3_corn3_22[nl[n]]);
	free(receive_E3_corn4_22[nl[n]]);
	#if(N3G>0)
	free(receive_E2_corn5_22[nl[n]]);
	free(receive_E2_corn6_22[nl[n]]);
	free(receive_E2_corn7_22[nl[n]]);
	free(receive_E2_corn8_22[nl[n]]);
	free(receive_E1_corn9_22[nl[n]]);
	free(receive_E1_corn10_22[nl[n]]);
	free(receive_E1_corn11_22[nl[n]]);
	free(receive_E1_corn12_22[nl[n]]);
	#endif
	#endif
	#endif
}

double calc_mem_cpu(int n)
{
	double mem=0.0;

	//Normal variables
	mem +=(NPRDUMP * BS_1*BS_2*BS_3 * sizeof(float));
	#if(DUMP_SMALL)
	mem +=(NPRDUMP * BS_1 / REDUCE_FACTOR1 * BS_2 / REDUCE_FACTOR2 * BS_3 / REDUCE_FACTOR3 * sizeof(float));
	mem +=(9 * BS_1 / REDUCE_FACTOR1 *BS_2 / REDUCE_FACTOR2 *BS_3 / REDUCE_FACTOR3 * sizeof(double));
	mem +=(49 * BS_1 / REDUCE_FACTOR1 *BS_2 / REDUCE_FACTOR2 * sizeof(double));
	#endif
	mem +=(9 * BS_1*BS_2*BS_3 * sizeof(double));
	mem +=(49 * BS_1*BS_2 * sizeof(double));
	mem +=((NPR + NDIM) * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(4 * BS_1*BS_2*BS_3 * sizeof(float));
	mem +=((BS_1 + 2 * N1G) * sizeof(double));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#if(DO_IMEX && RAD_M1)
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#else
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(double[NPR]));
	#endif
	#if(STAGGERED)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	#if(LEER)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double[6]));
	#endif
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPR]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int));
	#if(RAD_M1)
	mem +=((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G) * sizeof(int));
	#endif
	#if(CPU_OPENMP || 1)
	#if(STAGGERED)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[2][NDIM][NDIM]));
	#endif
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM]));
	#endif
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(int[NFAIL]));
	#if(!NSY)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NDIM][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPG]));
	#else
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NPG][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NPG]));
	#endif
	#if(FRAME_TRANSFORM)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double[NDIM][NDIM][NDIM]));
	#endif
	#if(ZIRI_DUMP)
	mem +=(BS_1 * BS_2 * BS_3 * 13 *sizeof(double));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)  * sizeof(double[NDIM][NDIM]));
	#endif
	#if (ELLIPTICAL2)
	mem +=((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double[NPR]));
	#endif
	
	//GPU transfer memory
	#if(GPU_ENABLED)
	mem +=(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * sizeof(double));
	mem +=(NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(double));
	mem +=(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * sizeof(double)); //array to store temporary data
	#if(STAGGERED)
	mem +=(NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * sizeof(double));
	mem +=(NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * sizeof(double));
	#endif
	#if(!NSY)
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*10 * sizeof(double));
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*10 * sizeof(double));
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NDIM*10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double));
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	mem +=(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG * sizeof(double));
	#else
	mem +=(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*10 * sizeof(double));
	mem +=(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*10 * sizeof(double));
	mem +=(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NDIM*10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	mem +=((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double));
	mem +=((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	mem +=(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * NPG  * sizeof(double));
	#endif
	mem +=(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * NFAIL * sizeof(int));
	#if(CARTESIAN_GR)
	mem += (((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G)) * sizeof(int));
	#endif
	mem += (((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G)) * sizeof(double));
	#endif

	//Boundaries
	int ref1_1, ref1_3, ref1_5, ref1_6;
	int ref2_2, ref2_4, ref2_5, ref2_6;
	int ref3_1, ref3_2, ref3_3, ref3_4;
	int ref1_1s, ref1_3s;
	int ref3_1s, ref3_3s, ref3_2s, ref3_4s;

	ref1_1 = REF_1; ref1_3 = REF_1; ref1_5 = REF_1; ref1_6 = REF_1;
	ref2_2 = REF_2; ref2_4 = REF_2; ref2_5 = REF_2; ref2_6 = REF_2;
	ref3_1 = REF_3; ref3_2 = REF_3; ref3_3 = REF_3; ref3_4 = REF_3;
	ref1_1s = REF_1; ref1_3s = REF_1;
	ref3_1s = REF_3; ref3_3s = REF_3;
	ref3_2s = REF_3;
	ref3_4s = REF_3;

	if (block[n][AMR_LEVEL] != N_LEVELS - 1){
		if (block[n][AMR_NBR1_3] >= 0) ref1_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR3_1] >= 0) ref1_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR1_3] >= 0) ref3_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_NBR3_1] >= 0) ref3_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
	}
	ref1_1s = ref1_1;
	ref1_3s = ref1_3;
	ref3_1s = ref3_1;
	ref3_3s = ref3_3;

	if (block[n][AMR_NBR1P] >= 0)ref1_1s = MY_MIN(ref1_1, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR1P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR3P] >= 0)ref1_3s = MY_MIN(ref1_3, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR3P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR1P] >= 0)ref3_1s = MY_MIN(ref3_1, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR1P]][AMR_LEVEL3]);
	if (block[n][AMR_NBR3P] >= 0)ref3_3s = MY_MIN(ref3_3, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR3P]][AMR_LEVEL3]);
	if ((block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*(int)pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1) && DEREFINE_POLE){
		ref3_2s = 0;
		ref3_4s = 0;
	}

	mem +=(NG *(1 + ref1_1)*(1 + ref3_1)* (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=(NG *(1 + ref2_2)*(1 + ref3_2)* (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	mem +=(NG *(1 + ref1_3)*(1 + ref3_3)* (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=(NG *(1 + ref2_4)*(1 + ref3_4)* (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=(NG *(1 + ref2_5)*(1 + ref1_5)* (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G) * (BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	mem +=(NG *(1 + ref2_6)*(1 + ref1_6)* (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G) * (BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	#endif

	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	#endif
	mem +=((1 + ref1_3)*(1 + ref3_3)* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_4)*(1 + ref3_4)* NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref1_1)*(1 + ref3_1)* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_2)*(1 + ref3_2)* NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=((1 + ref2_6)*(1 + ref1_6)* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	mem +=((1 + ref2_5)*(1 + ref1_5)* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	#endif
	#endif
	#if(PRESTEP==-100 || PRESTEP2==-100)
	mem +=((1 + ref1_3)*(1 + ref3_3)* 2* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_4)*(1 + ref3_4)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref1_1)*(1 + ref3_1)*2* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_2)*(1 + ref3_2)* 2*NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=((1 + ref2_6)*(1 + ref1_6)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	mem +=((1 + ref2_5)*(1 + ref1_5)*2* NG * (NPR + 3)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	mem +=(2*NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	#endif
	#endif
	#endif
	mem +=(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	mem +=((1 + ref1_3)*(1 + ref3_3)* (NPR)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_4)*(1 + ref3_4)* (NPR)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref1_1)*(1 + ref3_1)* (NPR)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_2)*(1 + ref3_2)* (NPR)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=((1 + ref2_6)*(1 + ref1_6)* (NPR)*(BS_2 / (1 + ref2_6) + 2 * N2G)*(BS_1 / (1 + ref1_6) + 2 * N1G) * sizeof(double));
	mem +=((1 + ref2_5)*(1 + ref1_5)* (NPR)*(BS_2 / (1 + ref2_5) + 2 * N2G)*(BS_1 / (1 + ref1_5) + 2 * N1G) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	#endif
	#if(CPU_OPENMP)
	mem +=(NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=(NPR*(BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	mem +=(NPR*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	mem +=(NPR* (BS_1)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_1)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	mem +=(NPR* (BS_2)*(BS_1) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	#endif
	mem +=(NPR* (BS_1)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_1)*(BS_3) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_3) * sizeof(double));
	#if(N3G>0)
	mem +=(NPR* (BS_2)*(BS_1) * sizeof(double));
	mem +=(NPR* (BS_2)*(BS_1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	mem +=(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4s)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	mem +=(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2s)) * sizeof(double));
	#if(N3G>0)
	mem +=(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_6)) *(BS_1 / (1 + ref1_6)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_5)) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	mem +=(NPR*(BS_2 / (1 + ref2_5) + 2 * N2G) *(BS_1 / (1 + ref1_5)) * sizeof(double));
	#endif
	#endif
	#endif
	#if(CPU_OPENMP || 1)
	mem +=(2 * (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(2 * (BS_2 + 2 * N2G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(2 * (BS_1 + 2 * N1G) *(BS_3 + 2 * N3G) * sizeof(double));
	mem +=(2 * (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=(2 * (BS_2 + 2 * N2G) *(BS_1 + 2 * N1G) * sizeof(double));
	mem +=(2 * (BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) * sizeof(double));
	#endif
	mem +=((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	#endif
	mem +=((1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * N3G) * sizeof(double));
	mem +=((1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * N3G) * sizeof(double));
	#if(N3G>0)
	mem +=((1 + ref2_6)*(1 + ref1_6) * 2 * (BS_2 / (1 + ref2_6) + 2 * D2)*(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=((1 + ref2_5)*(1 + ref1_5) * 2 * (BS_2 / (1 + ref2_5) + 2 * D2)*(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	#if(N3G>0)
	#endif
	mem +=(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2s) + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_6) + 2 * D2) *(BS_1 / (1 + ref1_6) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	mem +=(2 * (BS_2 / (1 + ref2_5) + 2 * D2) *(BS_1 / (1 + ref1_5) + 2 * D1) * sizeof(double));
	#endif
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#if(N_LEVELS>1)
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	mem +=((BS_3 + 2 * D3) * sizeof(double));
	#if(N3G>0)
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_2 + 2 * D2) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	mem +=((BS_1 + 2 * D1) * sizeof(double));
	#endif
	#endif
	#endif

	return mem;
}


double calc_mem_gpu(int n){
	int i;
	int ref1, ref2, ref3;
	double mem=0.;
	
	/*Allocate memory to buffers on GPU*/
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	#if(LEER)
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)0)*sizeof(double);
	#endif
	#if(CARTESIAN || CARTESIAN_GR)
	mem += ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G)) * sizeof(double);
	#else
	mem += (BS_1 + 2 * N1G)*sizeof(double);
	#endif
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	#if(DO_IMEX && (RAD_M1 || NEUTRINOS_M1))
	mem += NPR * ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G)) * sizeof(double);
	mem += NPR * ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G)) * sizeof(double);
	mem += NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(double);
	mem += NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(double);
	mem += NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(double);
	#endif
	#if((N_LEVELS_1D_INT>0) || RAD_M1 || RESISTIVE || TWO_T || NEUTRINOS_M1)
	mem += NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double); //Temp storage for conserved quantities
	mem += NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double); //Temp storage for cell centered electric field
	#endif
	#if(STAGGERED)
	mem += 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	mem += 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double);
	#endif
	#if(!NSY)
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*sizeof(double);
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG * 10 * sizeof(double);
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG * 10 * sizeof(double);
	#if(FRAME_TRANSFORM)
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double);
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double);
	#endif
	mem += ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NDIM * 10 * sizeof(double);
	#else
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG*sizeof(double);
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG * 10 * sizeof(double);
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NPG * 10 * sizeof(double);
	#if(FRAME_TRANSFORM)
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double);
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NSOLVER * NDIM * NDIM * sizeof(double);
	#endif
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*NDIM * 10 * sizeof(double);
	#endif
	mem += ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(int);
	#if(RAD_M1)
	mem += ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(int);
	#endif
	#if(CARTESIAN_GR)
	mem +=  ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(int);
	#endif
	mem +=((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) * NFAIL * sizeof(int);
	#if(NEUTRINOS_M1)
	mem += ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * sizeof(int);
	#if (NEUTRINOS_DEBUG)
	mem +=((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) * NEUTRINOS_DEBUG_NFLAGS * NU_SPECIES * sizeof(double);
	#endif
	#endif

	//Send buffers primitive variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5[0], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6[0], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5[0], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6[0], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend2_1[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Buffersend2_2[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Buffersend2_3[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Buffersend2_4[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend4_5[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Buffersend4_6[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Buffersend4_7[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Buffersend4_8[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend3_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Buffersend3_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Buffersend3_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Buffersend3_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend1_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Buffersend1_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Buffersend1_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Buffersend1_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)	
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend5_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Buffersend5_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Buffersend5_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Buffersend5_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Buffersend6_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Buffersend6_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Buffersend6_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Buffersend6_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers primitive variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec1[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec2[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec3[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec4[0], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec5[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec6[0], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec4_6[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec4_7[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec4_8[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec2_2[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec2_3[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec2_4[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4[0], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec1_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec1_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec1_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if(ref3)mem+=gpuMem(&Bufferrec1_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec3_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec3_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec3_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec6_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec6_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec6_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec5_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec5_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&Bufferrec5_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7[0], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	#if(PRESTEP || PRESTEP2)
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec1[0], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec2[0], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec3[0], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec4[0], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec5[0], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&tempBufferrec6[0], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec4_5[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec4_6[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec4_7[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec4_8[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec4_5[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&tempBufferrec4_6[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&tempBufferrec4_7[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&tempBufferrec4_8[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec2_1[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec2_2[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec2_3[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec2_4[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec2_1[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&tempBufferrec2_2[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&tempBufferrec2_3[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&tempBufferrec2_4[0], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec1_3[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec1_4[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec1_7[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec1_8[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec1_3[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&tempBufferrec1_4[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&tempBufferrec1_7[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&tempBufferrec1_8[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec3_1[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec3_2[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec3_5[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec3_6[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec3_1[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&tempBufferrec3_2[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&tempBufferrec3_5[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&tempBufferrec3_6[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec6_2[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec6_4[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec6_6[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec6_8[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec6_2[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&tempBufferrec6_4[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&tempBufferrec6_6[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&tempBufferrec6_8[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec5_1[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec5_3[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec5_5[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrec5_7[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&tempBufferrec5_1[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&tempBufferrec5_3[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&tempBufferrec5_5[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&tempBufferrec5_7[0], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif
	#endif

	//Send buffers flux variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5flux[0], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6flux[0], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5flux[0], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6flux[0], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers flux variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec1flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec2flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec3flux[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec4flux[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec5flux[0], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec6flux[0], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}	
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4flux[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7flux[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec1flux1[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec2flux1[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec3flux1[0], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec4flux1[0], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec5flux1[0], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec6flux1[0], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4flux1[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7flux1[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4flux2[0], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7flux2[0], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffers E variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5E[0], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6E[0], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend1E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend2E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend3E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend4E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend5E[0], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)mem+=gpuMem(&Buffersend6E[0], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers E variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec1E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec2E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec3E[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec4E[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec5E[0], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec6E[0], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4E[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7E[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec1E1[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec2E1[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec3E1[0], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec4E1[0], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec5E1[0], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)mem+=gpuMem(&Bufferrec6E1[0], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4E1[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7E1[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec4_5E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec4_6E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec4_7E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec4_8E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec2_1E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec2_2E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec2_3E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)mem+=gpuMem(&Bufferrec2_4E2[0], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec1_3E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec1_4E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec1_7E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec1_8E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec3_1E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)mem+=gpuMem(&Bufferrec3_2E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec3_5E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)mem+=gpuMem(&Bufferrec3_6E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec6_2E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec6_4E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec6_6E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec6_8E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		mem+=gpuMem(&Bufferrec5_1E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)mem+=gpuMem(&Bufferrec5_3E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)mem+=gpuMem(&Bufferrec5_5E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)mem+=gpuMem(&Bufferrec5_7E2[0], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn9[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn10[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn11[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn12[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN9P] >= 0 && block[block[n][AMR_CORN9P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn9[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10P] >= 0 && block[block[n][AMR_CORN10P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn10[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11P] >= 0 && block[block[n][AMR_CORN11P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn11[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12P] >= 0 && block[block[n][AMR_CORN12P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE1corn12[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5P] >= 0 && block[block[n][AMR_CORN5P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6P] >= 0 && block[block[n][AMR_CORN6P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7P] >= 0 && block[block[n][AMR_CORN7P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8P] >= 0 && block[block[n][AMR_CORN8P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE2corn8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN1P] >= 0 && block[block[n][AMR_CORN1P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2P] >= 0 && block[block[n][AMR_CORN2P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3P] >= 0 && block[block[n][AMR_CORN3P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4P] >= 0 && block[block[n][AMR_CORN4P]][AMR_ACTIVE] == 1) mem+=gpuMem(&BuffersendE3corn4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	//Receive buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE1corn11[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE1corn12[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE1corn9[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE1corn10[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE2corn7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE2corn8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE2corn5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE2corn6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE3corn3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE3corn4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE3corn1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) mem+=gpuMem(&BufferrecE3corn2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	
	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn11_2[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn11_6[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn12_4[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn12_8[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn9_3[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn9_7[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn10_1[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn10_5[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn7_5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn7_7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn8_6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn8_8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn5_2[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn5_4[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn6_3[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn6_1[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn3_5[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn3_6[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn4_7[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn4_8[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn1_3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn1_4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn2_1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn2_2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive tempbuffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE1corn11[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE1corn12[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE1corn9[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE1corn10[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE2corn7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE2corn8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE2corn5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE2corn6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE3corn3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE3corn4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE3corn1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) mem+=gpuMem(&tempBufferrecE3corn2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE1corn11_2[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE1corn11_6[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE1corn12_4[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE1corn12_8[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE1corn9_3[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE1corn9_7[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE1corn10_1[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE1corn10_5[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE2corn7_5[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE2corn7_7[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE2corn8_6[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE2corn8_8[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE2corn5_2[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE2corn5_4[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE2corn6_3[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE2corn6_1[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE3corn3_5[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE3corn3_6[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE3corn4_7[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE3corn4_8[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE3corn1_3[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE3corn1_4[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&tempBufferrecE3corn2_1[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&tempBufferrecE3corn2_2[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive buffer E2 corn
	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn11_22[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn11_62[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn12_42[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn12_82[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn9_32[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn9_72[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE1corn10_12[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE1corn10_52[0], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn7_52[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn7_72[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn8_62[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn8_82[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn5_22[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn5_42[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE2corn6_32[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE2corn6_12[0], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn3_52[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn3_62[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn4_72[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn4_82[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn1_32[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn1_42[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		mem+=gpuMem(&BufferrecE3corn2_12[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		mem+=gpuMem(&BufferrecE3corn2_22[0], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	return mem;
}

//Flag cells that need inflow boundary conditions in Cartesian mesh
void set_pflag_cart(int n) {
	int i, j, z;
	int offset;
	double delta_x1, delta_x2, delta_x3;
	int offset1, offset2, offset3;
	int flag;
	double X[NDIM], r, th, phi, rmin;

	//Calclate spacing of cells at event horizon
	delta_x1 = 2 * ROUT / (NB_1 * BS_1) / pow(1 + REF_1, block[n][AMR_LEVEL1]);
	delta_x2 = 2 * ROUT / (NB_2 * BS_2) / pow(1 + REF_2, block[n][AMR_LEVEL2]);
	delta_x3 = 2 * ROUT / (NB_3 * BS_3) / pow(1 + REF_3, block[n][AMR_LEVEL3]);

	//Calculate offset in number of cells
	offset1 = 1.2 / delta_x1;
	offset2 = 1.2 / delta_x2;
	offset3 = 1.2 / delta_x3;

	block[n][AMR_CARTFLAG] = 0;
	ZSLOOP3D(N1_GPU_offset[n] - N1G, BS_1 + N1_GPU_offset[n] + N1G-1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G-1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 + N3G-1) {	
		//Calculate coordiante
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);

		//Calculate rmin
		rmin = 0.8 * (1. + sqrt(1. - a * a));

		//Flag cells that are smaller than rmin
		if (r<rmin) {
			pflag_cart[nl[n]][index_3D(n, i, j, z)] = 1;
			block[n][AMR_CARTFLAG] = 1;
		}
		else {
			pflag_cart[nl[n]][index_3D(n, i, j, z)] = 0;
		}
	}
}

//Flag cells that need inflow boundary conditions in Spherical mesh
void set_pflag_rbound(int n) {
	int i, i2, j, z;
	double X[NDIM], r, th, phi;
	double rmin = RBOUND;

	/*
	Add flag, AMR_RBOUNDFLAG, which labels each block that contains cells that have cells marked for inflow boundary conditions
	*/
	block[n][AMR_RBOUNDFLAG] = 0;
	ZSLOOP3D(N1_GPU_offset[n] - N1G, BS_1 + N1_GPU_offset[n] + N1G-1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G-1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 + N3G-1) {	
		#if(CALC_METRIC)
		i2 = ((int)(i / pow(1 + REF_1, block[n][AMR_LEVEL1]))) * ((int)pow(1 + REF_1, block[n][AMR_LEVEL1])); //Making the index consistent near AMR boundaries
		coord(n, i2, j, z, FACE1, X);
		bl_coord(X, &r, &th, &phi);

		//Flag cells that are smaller than rmin
		if (r < (Rin*metric_scale_cpu)) {
			pflag_rbound[nl[n]][index_3D(n, i, j, z)] = 1;
			block[n][AMR_RBOUNDFLAG] = 1;
		}
		else {
			pflag_rbound[nl[n]][index_3D(n, i, j, z)] = 0;
		}
		#else
		//Calculate coordinate
		i2 = ((int)(i / pow(1 + REF_1, block[n][AMR_LEVEL1]))) * ((int)pow(1 + REF_1, block[n][AMR_LEVEL1])); //Making the index consistent near AMR boundaries
		coord(n, i2, j, z, FACE1, X);
		bl_coord(X, &r, &th, &phi);

		//Flag cells that are smaller than rmin
#if(NEUTRON_STAR)
		if (r < 1.01 * R_NS) {
#else
		if (r<(rmin - t/1000.0)) {
#endif
			pflag_rbound[nl[n]][index_3D(n, i, j, z)] = 1;
			block[n][AMR_RBOUNDFLAG] = 1;
		}
		else {
			pflag_rbound[nl[n]][index_3D(n, i, j, z)] = 0;
		}
		#endif

		#if(GPU_ENABLED)
		pflag_RBOUND_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = pflag_rbound[nl[n]][index_3D(n, i, j, z)];
		#endif
		#if(GPU_ENABLED && NEUTRON_STAR && !USE_PS1START)
		Bx1_surface_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = Bx1_surface[nl[n]][index_3D(n, i, j, z)];
		#endif
	}

	#if(GPU_ENABLED)
		#if(DO_RBOUND || (NEUTRON_STAR && 0))
		gpuMemcpyAsync(Bufferpflag_RBOUND[nl[n]], pflag_RBOUND_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
		#endif
	#endif
	#if(GPU_ENABLED && NEUTRON_STAR && !USE_PS1START)
		gpuMemcpyAsync(BufferBx1_surface[nl[n]], Bx1_surface_GPU[nl[n]], ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
}