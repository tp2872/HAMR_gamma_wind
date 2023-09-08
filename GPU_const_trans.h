__global__ void fix_flux(double *  F1, double *  F2, double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int icurr, jcurr, zcurr;
	int k;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id<(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		zcurr = global_id % (BS_3 + 2 * N3G);
		icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
		if (icurr >= N1G - D1 && zcurr >= N3G - D3 && icurr<BS_1 + N1G + D1 && zcurr<BS_3 + N3G + D3) {
			if (NBR_1 < 0){
				F1[B2*(ksize)+icurr*isize + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 9
				PLOOP F2[k*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr] = 0.;
				#endif
				#pragma unroll 9
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
			if (NBR_3 < 0){
				F1[B2*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 9
				PLOOP F2[k*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.;
				#endif
				#pragma unroll 9
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
		}
	}
	#if INFLOW==0
	else{
		global_id = global_id - (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
		zcurr = global_id % (BS_3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
		if (jcurr >= N2G - D2 && zcurr >= N3G - D3 && jcurr<BS_2 + N2G + D2 && zcurr<BS_3 + N3G + D3) {
			if (NBR_4<0){
				if (F1[RHO*(ksize)+N1G*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] > 0.) F1[RHO*(ksize)+N1G*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.;
			}
			if (NBR_2<0){
				if (F1[RHO*(ksize)+(BS_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] < 0.) F1[RHO*(ksize)+(BS_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.;
			}
		}

	}
	#endif
}

__global__ void consttransport1(const  double* __restrict__  pb_i, double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3 + 2 * D3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3);
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - D3)*D3;
	jcurr += (N2G - D2)*D2;
	icurr += (N1G - D1)*D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double pb[NPR];
	struct of_geom geom;

	if (k==1){
		for (k = 0; k<NPR; k++){
			pb[k] = pb_i[k*(ksize)+global_id];
		}
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		#if(RESISTIVE)
		double alpha, beta[3], E_cov[3];
		//Lapse in 3+1
		alpha = 1.0 / sqrt(-geom.gcon[0]);

		//Beta in 3+1
		beta[0] = geom.gcon[1] * alpha * alpha;
		beta[1] = geom.gcon[2] * alpha * alpha;
		beta[2] = geom.gcon[3] * alpha * alpha;

		/* dual of Maxwell tensor */
		lower_3(&(pb[E1]), geom.gcov, E_cov);

		//calculate the cell center values of the E-field
		#if(N3G>0)
		E_cent[1 * ksize + global_id] = geom.g * (beta[1] * pb[B3] - beta[2] * pb[B2] + (alpha * alpha / geom.g) * E_cov[0]); //-F2[B3], F3[B2]
		E_cent[2 * ksize + global_id] = geom.g * (beta[2] * pb[B1] - beta[0] * pb[B3] + (alpha * alpha / geom.g) * E_cov[1]); //-F3[B1], F1[B3]
		#endif
		E_cent[3 * ksize + global_id] = geom.g * (beta[0] * pb[B2] - beta[1] * pb[B1] + (alpha * alpha / geom.g) * E_cov[2]); //-F1[B2], F2[B1]
		#else
		struct of_state q;
		ucon_calc(pb, &geom, q.ucon);
		lower(q.ucon, geom.gcov, q.ucov);
		bcon_calc(pb, q.ucon, q.ucov, q.bcon);

		#if(N3G>0)
		E_cent[1 * ksize + global_id] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
		E_cent[2 * ksize + global_id] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
		#endif
		E_cent[3 * ksize + global_id] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);
		#endif
	}
}

__global__ void consttransport2(double *  emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2
	#if(CALC_MDOT)
	, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
	#if(DO_RBOUND)
	, int* pflag_rbound
	#endif
)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3 + D3)*(BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id<(BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int jsize = BS_3 + 2 * N3G;
	#if(CALC_MDOT)
	double factor = 1.0 / magnetic_density_scale;
	#else
	double factor = 1.0;
	#endif

	if (k==1){
		#if(RESISTIVE || CARTESIAN)
		double dE_LEFT_13_1 = 0.0;
		double dE_LEFT_13_2 = 0.0;
		double dE_RIGHT_13_1 = 0.0;
		double dE_RIGHT_13_2 = 0.0;
		double dE_LEFT_12_1 = 0.0;
		double dE_LEFT_12_2 = 0.0;
		double dE_RIGHT_12_1 = 0.0;
		double dE_RIGHT_12_2 = 0.0;
		double dE_LEFT_21_1 = 0.0;
		double dE_LEFT_21_2 = 0.0;
		double dE_RIGHT_21_1 = 0.0;
		double dE_RIGHT_21_2 = 0.0;
		double dE_LEFT_23_1 = 0.0;
		double dE_LEFT_23_2 = 0.0;
		double dE_RIGHT_23_1 = 0.0;
		double dE_RIGHT_23_2 = 0.0;
		double dE_LEFT_31_1 = 0.0;
		double dE_LEFT_31_2 = 0.0;
		double dE_RIGHT_31_1 = 0.0;
		double dE_RIGHT_31_2 = 0.0;
		double dE_LEFT_32_1 = 0.0;
		double dE_LEFT_32_2 = 0.0;
		double dE_RIGHT_32_1 = 0.0;
		double dE_RIGHT_32_2 = 0.0;
		#else
		double dE_LEFT_13_1 = E_cent[1 * (ksize)+global_id] - factor * F3[B2 * (ksize)+global_id];
		double dE_LEFT_13_2 = E_cent[1 * (ksize)+global_id - jsize * D2] - factor * F3[B2 * (ksize)+global_id - jsize * D2];
		double dE_RIGHT_13_1 = factor * F3[B2 * (ksize)+global_id + D3 - D3] - E_cent[1 * (ksize)+global_id - D3];
		double dE_RIGHT_13_2 = factor * F3[B2 * (ksize)+global_id + D3 - jsize * D2 - D3] - E_cent[1 * (ksize)+global_id - jsize * D2 - D3];
		double dE_LEFT_12_1 = E_cent[1 * (ksize)+global_id] + factor * F2[B3 * (ksize)+global_id];
		double dE_LEFT_12_2 = E_cent[1 * (ksize)+global_id - D3] + factor * F2[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_12_1 = -factor * F2[B3 * (ksize)+global_id + D2 * jsize - D2 * jsize] - E_cent[1 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_12_2 = -factor * F2[B3 * (ksize)+global_id + D2 * jsize - D2 * jsize - D3] - E_cent[1 * (ksize)+global_id - D2 * jsize - D3];
		double dE_LEFT_21_1 = E_cent[2 * (ksize)+global_id] - factor * F1[B3 * (ksize)+global_id];
		double dE_LEFT_21_2 = E_cent[2 * (ksize)+global_id - D3] - factor * F1[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_21_1 = factor * F1[B3 * (ksize)+global_id + D1 * isize - D1 * isize] - E_cent[2 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_21_2 = factor * F1[B3 * (ksize)+global_id + D1 * isize - D1 * isize - D3] - E_cent[2 * (ksize)+global_id - D1 * isize - D3];
		double dE_LEFT_23_1 = E_cent[2 * (ksize)+global_id] + factor * F3[B1 * (ksize)+global_id];
		double dE_LEFT_23_2 = E_cent[2 * (ksize)+global_id - D1 * isize] + factor * F3[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_23_1 = -factor * F3[B1 * (ksize)+global_id + D3 - D3] - E_cent[2 * (ksize)+global_id - D3];
		double dE_RIGHT_23_2 = -factor * F3[B1 * (ksize)+global_id + D3 - isize * D1 - D3] - E_cent[2 * (ksize)+global_id - isize * D1 - D3];
		double dE_LEFT_31_1 = E_cent[3 * (ksize)+global_id] + factor * F1[B2 * (ksize)+global_id];
		double dE_LEFT_31_2 = E_cent[3 * (ksize)+global_id - D2 * jsize] + factor * F1[B2 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_31_1 = -factor * F1[B2 * (ksize)+global_id + D1 * isize - D1 * isize] - E_cent[3 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_31_2 = -factor * F1[B2 * (ksize)+global_id + D1 * isize - D1 * isize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		double dE_LEFT_32_1 = E_cent[3 * (ksize)+global_id] - factor * F2[B1 * (ksize)+global_id];
		double dE_LEFT_32_2 = E_cent[3 * (ksize)+global_id - D1 * isize] - factor * F2[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_32_1 = factor * F2[B1 * (ksize)+global_id + D2 * jsize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_32_2 = factor * F2[B1 * (ksize)+global_id + D2 * jsize - D1 * isize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		#endif

		emf[1 * (ksize)+global_id] = 0.25*((-factor * F2[B3*(ksize)+global_id] - (dE_LEFT_13_1* (double)(F2[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_13_2* (double)(F2[RHO*(ksize)+global_id]>0.0)))
			+ (-factor * F2[B3*(ksize)+global_id - D3] + (dE_RIGHT_13_1* (double)(F2[RHO*(ksize)+global_id - D3] <= 0.0) + dE_RIGHT_13_2* (double)(F2[RHO*(ksize)+global_id - D3]>0.0))) +
			+(factor * F3[B2*(ksize)+global_id] - (dE_LEFT_12_1* (double)(F3[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_12_2* (double)(F3[RHO*(ksize)+global_id]>0.0)))
			+ (factor * F3[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_12_1* (double)(F3[RHO*(ksize)+global_id - D2*jsize] <= 0.0) + dE_RIGHT_12_2* (double)(F3[RHO*(ksize)+global_id - D2*jsize]>0.0))));
		emf[2 * (ksize)+global_id] = 0.25*((-factor * F3[B1*(ksize)+global_id] - (dE_LEFT_21_1* (double)(F3[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_21_2* (double)(F3[RHO*(ksize)+global_id]>0.0)))
			+ (-factor * F3[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_21_1* (double)(F3[RHO*(ksize)+global_id - D1*isize] <= 0.0) + dE_RIGHT_21_2* (double)(F3[RHO*(ksize)+global_id - D1*isize]>0.0)))
			+ (factor * F1[B3*(ksize)+global_id] - (dE_LEFT_23_1* (double)(F1[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_23_2* (double)(F1[RHO*(ksize)+global_id]>0.0)))
			+ (factor * F1[B3*(ksize)+global_id - D3] + (dE_RIGHT_23_1* (double)(F1[RHO*(ksize)+global_id - D3] <= 0.0) + dE_RIGHT_23_2* (double)(F1[RHO*(ksize)+global_id - D3]>0.0))));
		emf[3 * (ksize)+global_id] = 0.25*((factor * F2[B1*(ksize)+global_id] - (dE_LEFT_31_1* (double)(F2[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_31_2* (double)(F2[RHO*(ksize)+global_id]>0.0)))
			+ (factor * F2[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_31_1* (double)(F2[RHO*(ksize)+global_id - D1*isize] <= 0.0) + dE_RIGHT_31_2* (double)(F2[RHO*(ksize)+global_id - D1*isize]>0.0)))
			+ (-factor * F1[B2*(ksize)+global_id] - (dE_LEFT_32_1* (double)(F1[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_32_2* (double)(F1[RHO*(ksize)+global_id]>0.0)))
			+ (-factor * F1[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_32_1* (double)(F1[RHO*(ksize)+global_id - D2*jsize] <= 0.0) + dE_RIGHT_32_2* (double)(F1[RHO*(ksize)+global_id - D2*jsize] >0.0))));

		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == BS_2 + N2G)){
			emf[3 * (ksize)+global_id] = 0.;
			emf[1 * (ksize)+global_id] = -0.5 * factor * (F2[B3 * (ksize)+global_id] + F2[B3 * (ksize)+global_id - D3]);
		}

		#if(DO_RBOUND)
		if (pflag_rbound[global_id - isize * D1] == 1) {
			emf[3 * (ksize)+global_id] = 0.;
			emf[2 * (ksize)+global_id] = 0.;
		}
		#endif

		#if(CARTESIAN_GR)
		/*if (pflag_cart[global_id] == 1 || pflag_cart[global_id - D2 * jsize] == 1 || pflag_cart[global_id - D1 * isize] == 1 || pflag_cart[global_id - D1 * isize - D2 * jsize] == 1) {
			emf[3 * (ksize)+global_id] = 0.;
		}
		if (pflag_cart[global_id] == 1 || pflag_cart[global_id - D3] == 1 || pflag_cart[global_id - D1 * isize] == 1 || pflag_cart[global_id - D1 * isize + D3] == 1) {
			emf[2 * (ksize)+global_id] = 0.;
		}
		if (pflag_cart[global_id] == 1 || pflag_cart[global_id - D3] == 1 || pflag_cart[global_id - D2 * jsize] == 1 || pflag_cart[global_id - D2 * jsize + D3] == 1) {
			emf[1 * (ksize)+global_id] = 0.;
		}*/
		#endif
	}
}

__global__ void consttransport2_M1_2(double* emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2)
{
	/*int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + D3) * (BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr * (BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id < (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	int jsize = BS_3 + 2 * N3G;
	int zsize0 = 1, zsize1 = 1;
	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel0 = 0;
	int zoffset0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs((jcurr-D2) - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN((jcurr - D2) - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize0 = (int)(0.001 + pow(2.0, (double)zlevel0));
	zoffset0 = (zcurr - N3G) % zsize0;
	int zlevel1 = 0;
	int zoffset1;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize1 = (int)(0.001 + pow(2.0, (double)zlevel1));
	zoffset1 = (zcurr - N3G) % zsize1;
	#endif

	if (k == 1) {
		double dE_LEFT_13_1 = E_cent[1 * (ksize)+global_id] - F3[B2 * (ksize)+global_id];
		double dE_LEFT_13_2 = E_cent[1 * (ksize)+global_id - jsize * D2] - F3[B2 * (ksize)+global_id - jsize * D2];
		double dE_RIGHT_13_1 = F3[B2 * (ksize)+global_id ] - E_cent[1 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_13_2 = F3[B2 * (ksize)+global_id - jsize * D2] - E_cent[1 * (ksize)+global_id - jsize * D2 - D3 * zsize0];
		double dE_LEFT_12_1 = E_cent[1 * (ksize)+global_id] + F2[B3 * (ksize)+global_id];
		double dE_LEFT_12_2 = E_cent[1 * (ksize)+global_id - D3 * zsize1] + F2[B3 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_12_1 = -F2[B3 * (ksize)+global_id] - E_cent[1 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_12_2 = -F2[B3 * (ksize)+global_id - D3 * zsize0] - E_cent[1 * (ksize)+global_id - D2 * jsize - D3 * zsize0];
		double dE_LEFT_21_1 = E_cent[2 * (ksize)+global_id] - F1[B3 * (ksize)+global_id];
		double dE_LEFT_21_2 = E_cent[2 * (ksize)+global_id - D3] - F1[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_21_1 = F1[B3 * (ksize)+global_id] - E_cent[2 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_21_2 = F1[B3 * (ksize)+global_id - D3 * zsize1] - E_cent[2 * (ksize)+global_id - D1 * isize - D3*zsize1];
		double dE_LEFT_23_1 = E_cent[2 * (ksize)+global_id] + F3[B1 * (ksize)+global_id];
		double dE_LEFT_23_2 = E_cent[2 * (ksize)+global_id - D1 * isize] + F3[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_23_1 = -F3[B1 * (ksize)+global_id] - E_cent[2 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_23_2 = -F3[B1 * (ksize)+global_id - isize * D1] - E_cent[2 * (ksize)+global_id - isize * D1 - D3 * zsize1];
		double dE_LEFT_31_1 = E_cent[3 * (ksize)+global_id] + F1[B2 * (ksize)+global_id];
		double dE_LEFT_31_2 = E_cent[3 * (ksize)+global_id - D2 * jsize] + F1[B2 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_31_1 = -F1[B2 * (ksize)+global_id] - E_cent[3 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_31_2 = -F1[B2 * (ksize)+global_id - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		double dE_LEFT_32_1 = E_cent[3 * (ksize)+global_id] - F2[B1 * (ksize)+global_id];
		double dE_LEFT_32_2 = E_cent[3 * (ksize)+global_id - D1 * isize] - F2[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_32_1 = F2[B1 * (ksize)+global_id] - E_cent[3 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_32_2 = F2[B1 * (ksize)+global_id - D1 * isize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		
		emf[1 * (ksize)+global_id] *= 0.5;
		emf[2 * (ksize)+global_id] *= 0.5;
		emf[3 * (ksize)+global_id] *= 0.5;

		emf[1 * (ksize)+global_id] += 0.25 * 0.5 * ((-F2[B3 * (ksize)+global_id] - (dE_LEFT_13_1 * (double)(F2[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_13_2 * (double)(F2[RHO * (ksize)+global_id] > 0.0)))
			+ (-F2[B3 * (ksize)+global_id - D3] + (dE_RIGHT_13_1 * (double)(F2[RHO * (ksize)+global_id - D3] <= 0.0) + dE_RIGHT_13_2 * (double)(F2[RHO * (ksize)+global_id - D3] > 0.0))) +
			+(F3[B2 * (ksize)+global_id] - (dE_LEFT_12_1 * (double)(F3[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_12_2 * (double)(F3[RHO * (ksize)+global_id] > 0.0)))
			+ (F3[B2 * (ksize)+global_id - D2 * jsize] + (dE_RIGHT_12_1 * (double)(F3[RHO * (ksize)+global_id - D2 * jsize] <= 0.0) + dE_RIGHT_12_2 * (double)(F3[RHO * (ksize)+global_id - D2 * jsize] > 0.0))));
		emf[2 * (ksize)+global_id] += 0.25 * 0.5 * ((-F3[B1 * (ksize)+global_id] - (dE_LEFT_21_1 * (double)(F3[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_21_2 * (double)(F3[RHO * (ksize)+global_id] > 0.0)))
			+ (-F3[B1 * (ksize)+global_id - D1 * isize] + (dE_RIGHT_21_1 * (double)(F3[RHO * (ksize)+global_id - D1 * isize] <= 0.0) + dE_RIGHT_21_2 * (double)(F3[RHO * (ksize)+global_id - D1 * isize] > 0.0)))
			+ (F1[B3 * (ksize)+global_id] - (dE_LEFT_23_1 * (double)(F1[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_23_2 * (double)(F1[RHO * (ksize)+global_id] > 0.0)))
			+ (F1[B3 * (ksize)+global_id - D3] + (dE_RIGHT_23_1 * (double)(F1[RHO * (ksize)+global_id - D3] <= 0.0) + dE_RIGHT_23_2 * (double)(F1[RHO * (ksize)+global_id - D3] > 0.0))));
		emf[3 * (ksize)+global_id] += 0.25 * 0.5 * ((F2[B1 * (ksize)+global_id] - (dE_LEFT_31_1 * (double)(F2[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_31_2 * (double)(F2[RHO * (ksize)+global_id] > 0.0)))
			+ (F2[B1 * (ksize)+global_id - D1 * isize] + (dE_RIGHT_31_1 * (double)(F2[RHO * (ksize)+global_id - D1 * isize] <= 0.0) + dE_RIGHT_31_2 * (double)(F2[RHO * (ksize)+global_id - D1 * isize] > 0.0)))
			+ (-F1[B2 * (ksize)+global_id] - (dE_LEFT_32_1 * (double)(F1[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_32_2 * (double)(F1[RHO * (ksize)+global_id] > 0.0)))
			+ (-F1[B2 * (ksize)+global_id - D2 * jsize] + (dE_RIGHT_32_1 * (double)(F1[RHO * (ksize)+global_id - D2 * jsize] <= 0.0) + dE_RIGHT_32_2 * (double)(F1[RHO * (ksize)+global_id - D2 * jsize] > 0.0))));

		#if(!CARTESIAN)
		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == BS_2 + N2G)) {
			emf[3 * (ksize)+global_id] = 0.;
			emf[1 * (ksize)+global_id] += - 0.5 * (F2[B3 * (ksize)+global_id] + F2[B3 * (ksize)+global_id - D3]);
		}
		#endif
	}*/
}

__global__ void consttransport3(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0, i, imin[3], jmin[3], zmin[3], imax[3], jmax[3], zmax[3];
	isize = (BS_3 + D3)*(BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id<(BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	for (i = 0; i < 3; i++){
		imin[i] = N1G;
		jmin[i] = N2G;
		zmin[i] = N3G;
		imax[i] = BS_1 + N1G;
		jmax[i] = BS_2 + N2G;
		zmax[i] = BS_3 + N3G;
	}
	imax[0] += D1;
	jmax[1] += D2;
	zmax[2] += D3;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	#if(NSY)
	int index1 = FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	int index2 = FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	#if(N3G>0)
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	#endif
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#if(N3G>0)
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif
	#endif

	if (icurr >= imin[0] && jcurr >= jmin[0] && zcurr >= zmin[0] && icurr<imax[0] && jcurr<jmax[0]  && zcurr<zmax[0] && k==1){
		if (zoffset == 0){
			temp = 0.;
			for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[global_id - zoffset + u] * gdet_GPU[index1 - NSY*(zoffset - u)];
			for (u = 0; u < zsize; u++){
				temp += -Dt / ((double)zsize*dx_2)*(E_corn[3 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] - E_corn[3 * ksize + global_id - zoffset + u]) ;
			}
			#if(N3G>0)
			temp += Dt / ((double)zsize*dx_3)*(E_corn[2 * ksize + global_id - zoffset + D3*zsize] - E_corn[2 * ksize + global_id - zoffset]) ;
			#endif
			for (u = 0; u < zsize; u++)psf[global_id - zoffset + u] = temp / gdet_GPU[index1 - NSY*(zoffset - u)];
		}
	}

	if (icurr >= imin[2] && jcurr >= jmin[2] && zcurr >= zmin[2] && icurr<imax[2] && jcurr<jmax[2] && zcurr<zmax[2] && k == 1){
		#if(N3G>0)
		if (zoffset == 0){
			temp = psi[2 * ksize + global_id - zoffset] * gdet_GPU[index3 - NSY*(zoffset)];
			temp +=  - Dt / dx_1*(E_corn[2 * ksize + global_id + isize - zoffset] - E_corn[2 * ksize + global_id - zoffset]) ;
			temp += Dt / dx_2*(E_corn[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset] - E_corn[1 * ksize + global_id - zoffset]);
			if (zcurr == BS_3 + N3G) zsize = 1;
			for (u = 0; u < zsize; u++)psf[2 * ksize + global_id - zoffset + u] = temp / gdet_GPU[index3 - NSY*(zoffset-u)];
		}
		#endif
	}

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - D2 - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif
	if (icurr >= imin[1] && jcurr >= jmin[1] && zcurr >= zmin[1] && icurr<imax[1] && jcurr<jmax[1] && zcurr<zmax[1] && k == 1){
		if (zoffset == 0){
			temp = 0.;
			for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[1 * ksize + global_id - zoffset + u] * gdet_GPU[index2 - NSY*(zoffset - u)];
			for (u = 0; u < zsize; u++){
				temp += Dt / ((double)zsize*dx_1)*(E_corn[3 * ksize + global_id + isize - zoffset + u] - E_corn[3 * ksize + global_id - zoffset + u]) ;
			}
			#if(N3G>0)
			temp += -Dt / ((double)zsize*dx_3)*(E_corn[1 * ksize + global_id - zoffset + D3*zsize] - E_corn[1 * ksize + global_id - zoffset]);
			#endif
			for (u = 0; u < zsize; u++)psf[1 * ksize + global_id - zoffset + u] = temp / gdet_GPU[index2 - NSY*(zoffset - u)];
		}
	}
}

__global__ void consttransport3_post(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr, jcurr, zcurr, k=0, isize;
	if (global_id < (BS_2 + D2)*(BS_3 + D3)){
		k = 1;
		global_id -= 0;
		icurr = 0;
		zcurr = global_id%(BS_3 + D3);
		jcurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= (BS_2 + D2)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3)){
		k = 2;
		global_id -= (BS_2 + D2)*(BS_3 + D3);
		icurr = BS_1 - 1;
		zcurr = global_id%(BS_3 + D3);
		jcurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3)){
		k = 3;
		global_id -= 2 * (BS_2 + D2)*(BS_3 + D3);
		jcurr = 0;
		zcurr = global_id%(BS_3 + D3);
		icurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3)){
		k = 4;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3));
		jcurr = BS_2 - 1;
		zcurr = global_id%(BS_3 + D3);
		icurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2)){
		k = 5;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3));
		zcurr = 0;
		jcurr = global_id%(BS_2 + D2);
		icurr = (global_id - jcurr) / (BS_2 + D2);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_2 + D2)){
		k = 6;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2));
		zcurr = BS_3 - 1;
		jcurr = global_id%(BS_2 + D2);
		icurr = (global_id - jcurr) / (BS_2 + D2);
	}
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1,  zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	#if(NSY)
	int index1 = FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+(k==2))*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	int index2 = FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + (k == 4))*(BS_3 + 2 * N3G) + zcurr;
	#if(N3G>0)
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + (zcurr + (k==6));
	#endif
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + (k == 2))*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (jcurr + (k == 4));
	#if(N3G>0)
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif
	#endif

	if (k >= 1){
		if (icurr >= N1G + (k != 1) && jcurr >= N2G && zcurr >= N3G + (k == 3 || k == 4) && icurr < BS_1 + N1G + D1 - (k != 2) && jcurr < BS_2 + N2G  && zcurr < BS_3 + N3G - (k == 3 || k == 4)){
			if (zoffset == 0){
				temp = 0.;
				for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[global_id + (k == 2)*isize - zoffset + u] * gdet_GPU[index1 - NSY*(zoffset - u)];
				for (u = 0; u < zsize; u++){
					temp += -Dt / ((double)zsize*dx_2)*(E_corn[3 * ksize + global_id + (k == 2)*isize + (BS_3 + 2 * N3G) - zoffset + u] - E_corn[3 * ksize + global_id + (k == 2)*isize - zoffset + u]);
				}
				#if(N3G>0)
				temp += Dt / ((double)zsize*dx_3)*(E_corn[2 * ksize + global_id + (k == 2)*isize - zoffset + D3*zsize] - E_corn[2 * ksize + global_id + (k == 2)*isize - zoffset]);
				#endif
				for (u = 0; u < zsize; u++)psf[global_id + (k == 2)*isize - zoffset + u] = temp / gdet_GPU[index1 - NSY*(zoffset - u)];
			}
		}

		if (icurr >= N1G && jcurr >= N2G + (k == 1 || k == 2) && zcurr >= N3G + D3 * (k != 5) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G - (k == 1 || k == 2) && zcurr < BS_3 + N3G + D3 - (k != 6)){
			#if(N3G>0)
			if (zoffset == 0){
				temp = psi[2 * ksize + global_id - zoffset + (k == 6)] * gdet_GPU[index3 - NSY*(zoffset)];
				temp += -Dt / dx_1*(E_corn[2 * ksize + global_id + isize - zoffset + (k == 6)] - E_corn[2 * ksize + global_id - zoffset + (k == 6)]) ;
				temp += Dt / dx_2*(E_corn[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + (k == 6)] - E_corn[1 * ksize + global_id - zoffset + (k == 6)]);
				if (zcurr == BS_3 + N3G) zsize = 1;
				for (u = 0; u < zsize; u++)psf[2 * ksize + global_id - zoffset + u + (k == 6)] = temp / gdet_GPU[index3 - NSY*(zoffset-u)];
			}
			#endif
		}

		#if(N_LEVELS_1D_INT>0 && D3>0)
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (D2 + BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize = (int)(0.001+pow(2.0, (double)zlevel));
		zoffset = (zcurr - N3G) % zsize;
		#endif
		if (icurr >= N1G && jcurr >= N2G + (k != 3) && zcurr >= N3G + (k == 1 || k == 2) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G + D2 - (k != 4) && zcurr < BS_3 + N3G - (k == 1 || k == 2)){
			if (zoffset == 0){
				temp = 0.;
				for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u] * gdet_GPU[index2 - NSY*(zoffset - u)];
				for (u = 0; u < zsize; u++){
					temp += Dt / ((double)zsize*dx_1)*(E_corn[3 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) + isize - zoffset + u] - E_corn[3 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u]) ;
				}
				#if(N3G>0)
				temp += -Dt / ((double)zsize*dx_3)*(E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + D3*zsize] - E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset]);
				#endif
				for (u = 0; u < zsize; u++)psf[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u] = temp / gdet_GPU[index2 - NSY*(zoffset - u)];
			}
		}
	}
}

__global__ void flux_ct1(const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double *  emf)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + D3)*(BS_2 + D2);
	int zcurr = (global_id % (isize)) % (BS_3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	int icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G + D1 && jcurr<BS_2 + N2G + D2  && zcurr<BS_3 + N3G + D3){
		#if (N2G>0 && N3G>0)
		emf[1 * (ksize)+global_id] = -0.25*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id - 1] -
			F3[B2*(ksize)+global_id] - F3[B2*(ksize)+global_id - (BS_3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		emf[2 * (ksize)+global_id] = -0.25*(F3[B1*(ksize)+global_id] + F3[B1*(ksize)+global_id - isize] -
			F1[B3*(ksize)+global_id] - F1[B3*(ksize)+global_id - 1]);
		#endif
		#if (N1G>0 && N2G>0)
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (BS_3 + 2 * N3G)] -
			F2[B1*(ksize)+global_id] - F2[B1*(ksize)+global_id - isize]);
		#else
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (BS_3 + 2 * N3G)]);
		#endif
	}
}

__global__ void flux_ct2(double *  F1, double *  F2, double *  F3, const  double* __restrict__  emf)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + D3)*(BS_2 + D2);
	int zcurr = (global_id % (isize)) % (BS_3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	int icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double emf1 = -emf[1 * (ksize)+global_id];
	double emf2 = -emf[2 * (ksize)+global_id];
	double emf3 = -emf[3 * (ksize)+global_id];
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G + D1 && jcurr<BS_2 + N2G && zcurr<BS_3 + N3G){
		#if (N1G>0)
		F1[B1*(ksize)+global_id] = 0.0;
		#endif
		#if (N1G>0 && N2G>0)
		F1[B2*(ksize)+global_id] = 0.5*(emf3 - emf[3 * (ksize)+global_id + (BS_3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		F1[B3*(ksize)+global_id] = -0.5*(emf2 - emf[2 * (ksize)+global_id + 1]);
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G && jcurr<BS_2 + N2G + D2 && zcurr<BS_3 + N3G){
		#if (N1G>0 && N2G>0)
		F2[B1*(ksize)+global_id] = -0.5*(emf3 - emf[3 * (ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F2[B3*(ksize)+global_id] = 0.5*(emf1 - emf[1 * (ksize)+global_id + 1]);
		#endif
		#if(N2G>0)
		F2[B2*(ksize)+global_id] = 0.0;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G && jcurr<BS_2 + N2G && zcurr<BS_3 + N3G + D3){
		#if (N1G>0 && N3G>0)
		F3[B1*(ksize)+global_id] = 0.5*(emf2 - emf[2 * (ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F3[B2*(ksize)+global_id] = -0.5*(emf1 - emf[1 * (ksize)+global_id + (BS_3 + 2 * N3G)]);
		#endif
		#if(N3G>0)
		F3[B3*(ksize)+global_id] = 0.;
		#endif
	}
}
