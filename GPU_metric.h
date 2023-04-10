
/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ void raise(double ucov[NDIM], double gcon[10], double ucon[NDIM])
{
	#if AMD
	ucon[0] = fma(gcon[0], ucov[0], fma(gcon[1], ucov[1], fma(gcon[2], ucov[2], gcon[3] * ucov[3])));
	ucon[1] = fma(gcon[1], ucov[0], fma(gcon[4], ucov[1], fma(gcon[5], ucov[2], gcon[6] * ucov[3])));
	ucon[2] = fma(gcon[2], ucov[0], fma(gcon[5], ucov[1], fma(gcon[7], ucov[2], gcon[8] * ucov[3])));
	ucon[3] = fma(gcon[3], ucov[0], fma(gcon[6], ucov[1], fma(gcon[8], ucov[2], gcon[9] * ucov[3])));
	#else
	ucon[0] = gcon[0] * ucov[0] + gcon[1] * ucov[1] + gcon[2] * ucov[2] + gcon[3] * ucov[3];
	ucon[1] = gcon[1] * ucov[0] + gcon[4] * ucov[1] + gcon[5] * ucov[2] + gcon[6] * ucov[3];
	ucon[2] = gcon[2] * ucov[0] + gcon[5] * ucov[1] + gcon[7] * ucov[2] + gcon[8] * ucov[3];
	ucon[3] = gcon[3] * ucov[0] + gcon[6] * ucov[1] + gcon[8] * ucov[2] + gcon[9] * ucov[3];
	#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower(double ucon[NDIM], double gcov[10], double ucov[NDIM])
{
	#if AMD
	ucov[0] = fma(gcov[0], ucon[0], fma(gcov[1], ucon[1], fma(gcov[2], ucon[2], gcov[3] * ucon[3])));
	ucov[1] = fma(gcov[1], ucon[0], fma(gcov[4], ucon[1], fma(gcov[5], ucon[2], gcov[6] * ucon[3])));
	ucov[2] = fma(gcov[2], ucon[0], fma(gcov[5], ucon[1], fma(gcov[7], ucon[2], gcov[8] * ucon[3])));
	ucov[3] = fma(gcov[3], ucon[0], fma(gcov[6], ucon[1], fma(gcov[8], ucon[2], gcov[9] * ucon[3])));
	return;
	#else
	ucov[0] = gcov[0] * ucon[0] + gcov[1] * ucon[1] + gcov[2] * ucon[2] + gcov[3] * ucon[3];
	ucov[1] = gcov[1] * ucon[0] + gcov[4] * ucon[1] + gcov[5] * ucon[2] + gcov[6] * ucon[3];
	ucov[2] = gcov[2] * ucon[0] + gcov[5] * ucon[1] + gcov[7] * ucon[2] + gcov[8] * ucon[3];
	ucov[3] = gcov[3] * ucon[0] + gcov[6] * ucon[1] + gcov[8] * ucon[2] + gcov[9] * ucon[3];
	#endif
}

/* load local geometry into structure geom */
__device__ void get_geometry(int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU)
{
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jj*(BS_3 + 2 * N3G) + zz;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_2 + 2 * N2G) + jj;
	#endif
	#if(NSY)
	geom->gcon[0] = gcon_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[0] = gcov_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[1] = gcon_GPU[1 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[1] = gcov_GPU[1 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[2] = gcon_GPU[2 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[2] = gcov_GPU[2 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[3] = gcon_GPU[3 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[3] = gcov_GPU[3 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[4] = gcon_GPU[4 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[4] = gcov_GPU[4 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[5] = gcon_GPU[5 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[5] = gcov_GPU[5 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[6] = gcon_GPU[6 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[6] = gcov_GPU[6 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[7] = gcon_GPU[7 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[7] = gcov_GPU[7 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[8] = gcon_GPU[8 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[8] = gcov_GPU[8 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[9] = gcon_GPU[9 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[9] = gcov_GPU[9 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->g = gdet_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	#else
	geom->gcon[0] = gcon_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[0] = gcov_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[1] = gcon_GPU[1 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[1] = gcov_GPU[1 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[2] = gcon_GPU[2 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[2] = gcov_GPU[2 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[3] = gcon_GPU[3 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[3] = gcov_GPU[3 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[4] = gcon_GPU[4 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[4] = gcov_GPU[4 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[5] = gcon_GPU[5 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[5] = gcov_GPU[5 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[6] = gcon_GPU[6 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[6] = gcov_GPU[6 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[7] = gcon_GPU[7 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[7] = gcov_GPU[7 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[8] = gcon_GPU[8 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[8] = gcov_GPU[8 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[9] = gcon_GPU[9 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[9] = gcov_GPU[9 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->g = gdet_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	#endif
}