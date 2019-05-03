#include <cuda.h>
#include "config.h"

__device__ double slope_lim(double y1, double y2, double y3);

__device__ double slope_lim(double y1, double y2, double y3)
{
	double Dqm, Dqp, Dqc, s;
	/* woodward, or monotonized central, slope limiter */
	Dqm = (2.0)*(y2 - y1);
	Dqp = (2.0)*(y3 - y2);
	Dqc = 0.5*(y3 - y1);
	s = Dqm*Dqp;
	if (s <= 0.) return 0.;
	else {
		if (fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
			return(Dqm);
		else if (fabs(Dqp) < fabs(Dqc))
			return(Dqp);
		else
			return(Dqc);
	}
}

__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (i = i1; i < i2; i++){
				send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		#if(STAGGERED)
		for (i = i1; i <i2; i++){
			send[(NPR + 0)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+(i + N1G + (i1>N1G))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			send[(NPR + 1)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			send[(NPR + 2)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		#endif
	}
}

__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (j = j1; j < j2; j++){
				send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		
		#if(STAGGERED)
		for (j = j1; j <j2; j++){
			send[(NPR + 0)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			send[(NPR + 1)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+icurr*isize + (j + N2G + (j1>N2G))*(BS_3 + 2 * N3G) + zcurr];
			send[(NPR + 2)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
		#endif
	}
}

__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int POLE_1, int POLE_2)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;	
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel = 0;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	#endif

	if (global_id < work_size){
		if (z1 < BS_3 / 2){
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z*zsize + N3G)];
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + BS_3 + N3G - D3*zsize - (BS_3 - D3 - z)*zsize];
				}
			}
		}

		#if(STAGGERED)
		for (z = z1; z <z2; z++){
			send[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			send[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			send[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G))];
		}
		#endif
	}
}

__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (i = i1; i < i2; i += 1 + ref_1){
				send[k*jsize2*zsize2*(i2 - i1) / (1 + ref_1) + (i - i1) / (1 + ref_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = 0.125*(
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(i + ref_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + ref_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3] +
					pv[k*(ksize)+(i + ref_1 + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + ref_1 + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
		#if(STAGGERED)
		for (i = i1; i <i2; i += 1 + ref_1){
			send[(NPR + 0)*jsize2*zsize2*(i2 - i1) / (1 + ref_1) + (i - i1) / (1 + ref_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + ref_1))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] +
				ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + ref_1))*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + ref_3)] +
				ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + ref_1))*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (zcurr)] +
				ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + ref_1))*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);

			send[(NPR + 1)*jsize2*zsize2*(i2 - i1) / (1 + ref_1) + (i - i1) / (1 + ref_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] +
				ps[1 * (ksize)+(i + N1G)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + ref_3)] +
				ps[1 * (ksize)+(i + N1G + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] +
				ps[1 * (ksize)+(i + N1G + ref_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);

			send[(NPR + 2)*jsize2*zsize2*(i2 - i1) / (1 + ref_1) + (i - i1) / (1 + ref_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] +
				ps[2 * (ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr] +
				ps[2 * (ksize)+(i + N1G + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] +
				ps[2 * (ksize)+(i + N1G + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (zcurr)]);
		}
		#endif
	}
}

__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (j = j1; j < j2; j += 1 + ref_2){
				send[k*isize2*zsize2*(j2 - j1) / (1 + ref_2) + (j - j1) / (1 + ref_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = 0.125*(
					pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(icurr)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(icurr)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3] +
					pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
		#if(STAGGERED)
		for (j = j1; j <j2; j += 1 + ref_2){
			send[(NPR + 0)*isize2*zsize2*(j2 - j1) / (1 + ref_2) + (j - j1) / (1 + ref_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] +
				ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr + ref_3)] +
				ps[0 * (ksize)+(icurr)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + (zcurr)] +
				ps[0 * (ksize)+(icurr)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);

			send[(NPR + 1)*isize2*zsize2*(j2 - j1) / (1 + ref_2) + (j - j1) / (1 + ref_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + ref_2))*(BS_3 + 2 * N3G) + zcurr] +
				ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr + ref_3)] +
				ps[1 * (ksize)+(icurr + ref_1)*isize + (j + N2G + (j1>N2G)*(1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr)] +
				ps[1 * (ksize)+(icurr + ref_1)*isize + (j + N2G + (j1>N2G)*(1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);

			send[(NPR + 2)*isize2*zsize2*(j2 - j1) / (1 + ref_2) + (j - j1) / (1 + ref_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] =
				0.25*(ps[2 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + 
				ps[2 * (ksize)+(icurr)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + zcurr]  + 
				ps[2 * (ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr)] + 
				ps[2 * (ksize)+(icurr + ref_1)*isize + (j + N2G + ref_2)*(BS_3 + 2 * N3G) + (zcurr)]);
		}
		#endif
	}
}

__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + ref_2))*(1 + ref_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + ref_2))) / ((j2 - j1) / (1 + ref_2))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (z = z1; z < z2; z += 1 + ref_3){
				send[k*isize2*jsize2*(z2 - z1) / (1 + ref_3) + (z - z1) / (1 + ref_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] = 0.125*(
					pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + ref_3] + pv[k*(ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G] +
					pv[k*(ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G + ref_3] + pv[k*(ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + ref_3] +
					pv[k*(ksize)+(icurr + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G + ref_3]);
			}
		}
		#if(STAGGERED)
		for (z = z1; z <z2; z += 1 + ref_3){
			send[(NPR + 0)*isize2*jsize2*(z2 - z1) / (1 + ref_3) + (z - z1) / (1 + ref_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] =
				0.25*(ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] +
				ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + ref_3)] +
				ps[0 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G] +
				ps[0 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (z + N3G + ref_3)] );

			send[(NPR + 1)*isize2*jsize2*(z2 - z1) / (1 + ref_3) + (z - z1) / (1 + ref_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] =
				0.25*(ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] +
				ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + ref_3)] +
				ps[1 * (ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] +
				ps[1 * (ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + ref_3)]);

			send[(NPR + 2)*isize2*jsize2*(z2 - z1) / (1 + ref_3) + (z - z1) / (1 + ref_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] =
				0.25*(ps[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + ref_3))] +
				ps[2 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + ref_3))] +
				ps[2 * (ksize)+(icurr + ref_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + ref_3))] +
				ps[2 * (ksize)+(icurr + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + ref_3))]);
		}
		#endif
	}
}

__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		#if(PRESTEP2)
		//When at last timestep store old value
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR+3; k++){
				for (i = i1; i < i2; i++){
					tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
						= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				}
			}
		}

		//When at first timestep for interpolation calculate the gradient in time
		if (nstep%timelevel_rec == timelevel - 1   || nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				for (i = i1; i < i2; i++){
					tempreceive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
						= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] 
						- tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]) / (0.5*dt*timelevel_rec);
				}
			}
		}
		#elif(PRESTEP==-100)
		//When at last timestep store old value
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				for (i = i1; i < i2; i++){
					tempreceive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
						= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] -
						tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]) / (0.5*dt*timelevel_rec);
					tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
						= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
					if (nstep == -1) tempreceive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] = 0.;
				}
			}
		}
		#endif
		//When at subsequent timesteps for interpolation
		if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && jcurr >= N2G && jcurr<BS_2 + N2G && zcurr >= N3G && zcurr < BS_3 + N3G) {
			#if(PRESTEP==-100 || PRESTEP2)
			for (k = 0; k < NPR; k++){
				for (i = i1; i < i2; i++){
					p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				}
			}
			for (i = i1; i < i2; i++){
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (tempreceive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
					((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(1 + NPR + NPR + 3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (tempreceive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
					((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(2 + NPR + NPR + 3)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			#endif
		}

		//Reset primitve variables
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (i = i1; i < i2; i++){
					p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				}
			}
			#if(STAGGERED)
			for (i = i1; i <i2; i++){
				//if ((jcurr<N2G || jcurr >= BS_2 + N2G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
				//	ps[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				//	psh[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				//}
				//else psh[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * 2 * D1 / (1 + ref_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
			}
			#endif
		}
	}
}

__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;	
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;
	if (global_id < work_size){
		if (reverse == 0){
			#if(PRESTEP2)
			//When at last timestep store old value
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				for (k = 0; k < NPR + 3; k++){
					for (j = j1; j < j2; j++){
						tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
							= receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					}
				}
			}

			//When at first timestep for interpolation calculate the gradient in time
			if (nstep%timelevel_rec == timelevel - 1  || nstep == -1){
				for (k = 0; k < NPR + 3; k++){
					for (j = j1; j < j2; j++){
						tempreceive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
							= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] - 
							tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]) / (0.5*dt*timelevel_rec);
					}
				}
			}
			#elif(PRESTEP==-100)
			//When at last timestep store old value
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				for (k = 0; k < NPR + 3; k++){
					for (j = j1; j < j2; j++){
						tempreceive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
							= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] -
							tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]) / (0.5*dt*timelevel_rec);
						tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]
							= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
						if (nstep == -1) tempreceive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] = 0.;
					}
				}
			}
			#endif
			//When at subsequent timesteps for interpolation
			if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && icurr >= N1G && icurr<BS_1 + N1G && zcurr >= N3G && zcurr < BS_3 + N3G) {
				#if(PRESTEP==-100 || PRESTEP2)
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]+
							((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
						ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]+
							((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					}
				}
				for (j = j1; j < j2; j++){
					ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (tempreceive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(0 + NPR + NPR + 3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
					psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
					ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (tempreceive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] +
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(2 + NPR + NPR + 3)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))]);
					psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
				#endif
			}

			//Reset primitve variables
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				for (k = 0; k < NPR; k++){
					//#pragma unroll NG
					for (j = j1; j < j2; j++){
						p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
						ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					}
				}
				for (j = j1; j <j2; j++){
					ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					//if ((icurr<N1G || icurr >= BS_1 + N1G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
					//	ps[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					//	psh[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					//}
					//else psh[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * 2 * D2 / (1 + ref_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (j = j1; j < j2; j++){
					if (k == 3 || k == 4 || k == 6 || k == 7) factor = -1.;
					else factor = 1.;
					p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
					ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				}
			}
			#if(STAGGERED)
			for (j = j1; j <j2; j++){
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))] ;
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];			
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*zsize2 + (zcurr - z1 - N3G + z_offset * 2 * D3 / (1 + ref_3))];
			}
			#endif
		}
	}
}

__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		#if(PRESTEP2)
		//When at last timestep store old value
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				for (z = z1; z < z2; z++){
					tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]);
				}
			}
		}

		//When at first timestep for interpolation calculate the gradient in time
		if (nstep%timelevel_rec == timelevel - 1  || nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				for (z = z1; z < z2; z++){
					tempreceive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))] - 
						tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]) / (0.5*dt*timelevel_rec);
				}
			}
		}
		#elif(PRESTEP==-100)
		//When at last timestep store old value
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				for (z = z1; z < z2; z++){
					tempreceive[(k + NPR + 3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))] -
						tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]) / (0.5*dt*timelevel_rec);
					tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]);	
					if(nstep==-1) tempreceive[(k + NPR + 3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))] = 0.;		
				}
			}
		}
		#endif
		//When at subsequent timesteps for interpolation
		if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && icurr >= N1G && icurr<BS_1 + N1G && jcurr >= N2G && jcurr < BS_2 + N2G) {
			#if(PRESTEP==-100 || PRESTEP2)
			for (k = 0; k < NPR; k++){
				for (z = z1; z < z2; z++){
					p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = tempreceive[(k)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]+
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
					ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = tempreceive[(k)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]+
						((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				}
			}
			for (z = z1; z < z2; z++){
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (tempreceive[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))] +
					((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(NPR + 0 + NPR + 3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]);
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (tempreceive[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))] +
					((nstep + 1) % timelevel_rec)*0.5*dt*(double)timelevel*tempreceive[(NPR + 1 + NPR + 3)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))]);
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
			#endif
		}

		//Reset primitive variables
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (z = z1; z < z2; z++){
					p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
					ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				}
			}
			#if(STAGGERED)
			for (z = z1; z <z2; z++){
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				//if ((icurr<N1G || icurr >= BS_1 + N1G || jcurr<N2G || jcurr >= BS_2 + N2G) && update_staggered == 1){
				//	ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				//	psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];
				//}
				//else psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * 2 * D3 / (1 + ref_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * 2 * D1 / (1 + ref_1))*jsize2 + (jcurr - j1 - N2G + j_offset * 2 * D2 / (1 + ref_2))];			
			}
			#endif
		}
	}
}
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int i, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR + 3], dq1[NPR + 3], dq2[NPR + 3], dq3[NPR + 3];
	double receive_local[(NPR + 3)*NG*(1 + 2 * REF_2)*(1 + 2 * REF_3)];
	int ii1, ij1, iz1;

	if (global_id < work_size){
		//#pragma unroll NG
		for (i = i1; i < i2; i++){
			if (i1 < 0 && ref_1 == 1) ii = (NG - 1) + (i + 1) / (1 + ref_1);
			else if (ref_1 == 1) ii = (i - BS_1) / (1 + ref_1);
			else ii = i - i1;
			ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + ref_2)) / (1 + ref_2) + ref_2;
			iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + ref_3)) / (1 + ref_3) + ref_3;

			if (i < 0){
				if (i == -3) is = 1;
				else if (i == -2) is = -1;
				else if (i == -1) is = 1;
			}
			if (i>0){
				if (i == BS_1) is = -1;
				else if (i == BS_1 + 1) is = 1;
				else if (i == BS_1 + 2) is = -1;
			}
			js = (((jcurr - j1 - N2G) % (1 + ref_2) == 0) ? (-1) : (1));
			zs = (((zcurr - z1 - N3G) % (1 + ref_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR + 3; k++){
				for (ii1 = 0; ii1 < i2 - i1; ii1++)for (ij1 = ij - ref_2; ij1 <= ij + ref_2; ij1++)for (iz1 = iz - ref_3; iz1 <= iz + ref_3; iz1++){
					receive_local[k*(i2 - i1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + ii1*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij1 - (ij - ref_2))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
					#if(PRESTEP2==100)
					//Store at full step the value at t in temp1
					if (nstep%timelevel_rec == timelevel_rec - 1 || nstep==-1){
						temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] =  receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
					}
					//Store at prestep the time gradient in temp2
					if (nstep%timelevel_rec == timelevel - 1 || nstep%timelevel_rec == timelevel_rec - 1 || nstep==-1){
						temp1receive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] = (receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1]-temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1])/(timelevel_rec*0.5*dt);
					}
					//Add gradient to boundary
					if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && zcurr >= N3G && zcurr<BS_3 + N3G && jcurr >= N2G && jcurr < BS_2 + N2G) {
						receive_local[k*(i2 - i1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + ii1*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij1 - (ij - ref_2))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] + (nstep+1)%timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
					}
					#elif(PRESTEP==-100)
					if(nstep%timelevel_rec==timelevel_rec-1 || nstep == -1){ 
						temp1receive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] = (receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1]-temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1])/(timelevel_rec*0.5*dt);
					}

					if (nstep == -1 || timelevel_rec <= timelevel){
						temp1receive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] = 0.0;
						temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
					}

					if (nstep != -1 && timelevel_rec>timelevel && nstep%timelevel_rec!=timelevel_rec-1){
						temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
						receive_local[k*(i2 - i1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + ii1*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij1 - (ij - ref_2))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1] + (nstep + 1) % timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*jsize2*zsize2*(i2 - i1) + ii1*jsize2*zsize2 + ij1*zsize2 + iz1];
					}
					#endif
				}
			}
			ij = ref_2;
			iz = ref_3;

			for (k = 0; k < NPR; k++){
				dq1[k] = 0.0;
				if (ref_1){
					if (ii == 0 || ii == NG - 1){
						avg[k] = 0.125*(prim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] + prim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[k] += 0.125*(prim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] + prim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[k] += 0.125*(prim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)] + prim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)]);
						avg[k] += 0.125*(prim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)] + prim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)]);
						if (ii == 0){
							dq1[k] = slope_lim(avg[k], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)* (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)]);
						}
						else{
							dq1[k] = slope_lim(receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii - 1) * (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], avg[k]);
						}
					}
					else{
						dq1[k] = slope_lim(receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii - 1) * (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii+1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)]);
					}
				}
				dq2[k] = slope_lim(receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij - ref_2)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij + ref_2)*(1 + 2 * ref_3) + (iz)]);
				dq3[k] = slope_lim(receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz - ref_3)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz + ref_3)]);
			}
			for (k = 0; k < 3; k++){
				dq1[(k + NPR)] = 0.0;
				if (ref_1){
					if (ii == 0 || ii == NG - 1){
						avg[(k + NPR)] = 0.25*(psim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] +
							psim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[(k + NPR)] += 0.25*(psim[k*(ksize)+(N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2*(k == 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3*(k == 1))] +
							psim[k*(ksize)+(ii / (NG - 1) + N1G + (NG - 1 - ii) / (NG - 1)*(BS_1 - 2))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2*(k == 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3*(k == 1))]);
						if (ii == 0){
							dq1[(k + NPR)] = slope_lim(avg[(k + NPR)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)* (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)]);
						}
						else{
							dq1[(k + NPR)] = slope_lim(receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii - 1) * (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], avg[(k + NPR)]);
						}
					}
					else{
						dq1[(k + NPR)] = slope_lim(receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii - 1) * (1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + (ii+1)*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)]);
					}
				}
				dq2[(k + NPR)] = slope_lim(receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij - ref_2)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij + ref_2)*(1 + 2 * ref_3) + (iz)]);
				dq3[(k + NPR)] = slope_lim(receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz - ref_3)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)], receive_local[(k + NPR)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz + ref_3)]);
			}
			//for (k = 0; k < NPR + 3; k++) dq1[k] = dq2[k] = dq3[k] = 0.;
			for (k = 0; k < NPR; k++){
				p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
				ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive_local[k*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
			}
			//for (k = 0; k < NPR+3; k++) dq1[k] = dq2[k] = dq3[k] = 0.;

			#if(STAGGERED)
			if (js == 1){
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + ((ij)+ref_2)*(1 + 2 * ref_3) + (iz)])
					+ 0.25*(double)(is)*dq1[B2] * ref_1 + 0.25*(double)(zs)*dq3[B2] * ref_3; 
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + ((ij)+ref_2)*(1 + 2 * ref_3) + (iz)])
					+ 0.25*(double)(is)*dq1[B2] * ref_1 + 0.25*(double)(zs)*dq3[B2] * ref_3;
			}
			else{
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[NPR + 1] * ref_1 + 0.25*(double)(zs)*dq3[NPR + 1] * ref_3);
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 1)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[NPR + 1] * ref_1 + 0.25*(double)(zs)*dq3[NPR + 1] * ref_3);
			}

			if (zs == 1){
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz + ref_3)])
					+ 0.25*(double)(is)*dq1[B3] * ref_1 + 0.25*(double)(js)*dq2[B3] * ref_2;
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz + ref_3)])
					+ 0.25*(double)(is)*dq1[B3] * ref_1 + 0.25*(double)(js)*dq2[B3] * ref_2;
			}
			else{
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[NPR + 2] * ref_1 + 0.25*(double)(js)*dq2[NPR + 2] * ref_2);
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 2)*(1 + 2 * ref_2)*(1 + 2 * ref_3)*(i2 - i1) + ii*(1 + 2 * ref_2)*(1 + 2 * ref_3) + (ij)*(1 + 2 * ref_3) + (iz)] + 0.25*(double)(is)*dq1[NPR + 2] * ref_1 + 0.25*(double)(js)*dq2[NPR + 2] * ref_2);
			}
			#endif
		}
	}
}

__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int j, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;	
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR + 3], dq1[NPR + 3], dq2[NPR + 3], dq3[NPR + 3];
	double receive_local[(NPR + 3)*NG*(1 + 2 * REF_1)*(1 + 2 * REF_3)];
	int ii1, ij1, iz1;

	if (global_id < work_size){
//#pragma unroll NG
		for (j = j1; j < j2; j++){
			if (j1 < 0 && ref_2 == 1) ij = (NG - 1) + (j + 1) / (1 + ref_2);
			else if (ref_2 == 1) ij = (j - BS_2) / (1 + ref_2);
			else ij = j - j1;

			ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + ref_1)) / (1 + ref_1) + ref_1;
			iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + ref_3)) / (1 + ref_3) + ref_3;

			is = (((icurr - i1 - N1G) % (1 + ref_1) == 0) ? (-1) : (1));
			if (j < 0){
				if (j == -3) js = 1;
				else if (j == -2) js = -1;
				else if (j == -1) js = 1;
			}
			if (j>0){
				if(j == BS_2) js=-1;
				else if (j == BS_2 + 1) js = 1;
				else if (j == BS_2 + 2) js = -1;
			}	
			zs = (((zcurr - z1 - N3G) % (1 + ref_3) == 0) ? (-1) : (1));

			for (k = 0; k < NPR+3; k++){
				for (ii1 = ii-ref_1; ii1 <= ii+ref_1; ii1++)for (ij1 = 0; ij1 < j2-j1; ij1++)for (iz1 = iz - ref_3; iz1 <= iz + ref_3; iz1++){
					receive_local[k*(j2 - j1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ij1*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii1 - (ii - ref_1))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
					#if(PRESTEP2==100)
					//Store at full step the value at t in temp1
					if (nstep%timelevel_rec == timelevel_rec - 1 || nstep==-1){
						temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] =  receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
					}
					//Store at prestep the time gradient in temp2
					if (nstep%timelevel_rec == timelevel - 1 || nstep%timelevel_rec == timelevel_rec - 1 || nstep==-1){
						temp1receive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] = (receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1]-temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1])/(timelevel_rec*0.5*dt);
					}
					//Add gradient to boundary
					if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && icurr >= N1G && icurr<BS_1 + N1G && zcurr >= N3G && zcurr < BS_3 + N3G) {
						receive_local[k*(j2 - j1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ij1*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii1 - (ii - ref_1))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] + (nstep + 1) % timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
					}
					#elif(PRESTEP==-100)
					if(nstep%timelevel_rec==timelevel_rec-1 || nstep == -1){ 
						temp1receive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] = (receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1]-temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1])/(timelevel_rec*0.5*dt);
					}
					if (nstep == -1 || timelevel_rec <= timelevel){
						temp1receive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] = 0.0;
						temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] = receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
					}

					if (nstep != -1 && timelevel_rec>timelevel && nstep%timelevel_rec!=timelevel_rec-1){
						temp1receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] = receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
						receive_local[k*(j2 - j1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ij1*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii1 - (ii - ref_1))*(1 + 2 * ref_3) + (iz1 - (iz - ref_3))] = receive[k*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1] + (nstep + 1) % timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*isize2*zsize2*(j2 - j1) + ij1*isize2*zsize2 + ii1*zsize2 + iz1];
					}
					#endif
				}
			}
			ii = ref_1;
			iz = ref_3;
			for (k = 0; k < NPR; k++){
				dq2[k] = 0.0;
				if (ref_2){
					if (ij == 0 || ij == NG - 1){
						avg[k] = 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3)]);
						if (ij == 0){
							dq2[k] = slope_lim(avg[k], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)* (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)]);
						}
						else{
							dq2[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], avg[k]);
						}
					}
					else{
						dq2[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)]);
					}
				}
				dq1[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii - ref_1)*(1 + 2 * ref_3) + (iz)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + iz], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii + ref_1)*(1 + 2 * ref_3) + (iz)]);
				dq3[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + (iz - ref_3)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + iz], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + (iz + ref_3)]);
			}
			for (k = 0; k < 3; k++){
				dq2[(NPR + k)] = 0.0;
				if (ref_2){
					if (ij == 0 || ij == NG - 1){
						avg[NPR + k] = 0.25*(psim[k * (ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))] +
							psim[k * (ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3))]);
						avg[NPR + k] += 0.25*(psim[k * (ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1*(k == 2))*isize + (N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 1))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3*(k == 0))] +
							psim[k * (ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1*(k == 2))*isize + (ij / (NG - 1) + N2G + (NG - 1 - ij) / (NG - 1)*(BS_2 - 2))*(BS_3 + 2 * N3G) + (zcurr - (zcurr - N3G) % (1 + ref_3) + ref_3*(k == 0))]);
						if (ij == 0){
							dq2[NPR + k] = slope_lim(avg[NPR + k], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)* (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)]);
						}
						else{
							dq2[NPR + k] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], avg[(NPR + k)]);
						}
					}
					else{
						dq2[NPR + k] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + (ij+1)*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz)]);
					}
				}
				dq1[(NPR + k)] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii - ref_1)*(1 + 2 * ref_3) + (iz)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + iz], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii + ref_1)*(1 + 2 * ref_3) + (iz)]);
				dq3[(NPR + k)] =  slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + (iz - ref_3)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + iz], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii)*(1 + 2 * ref_3) + (iz + ref_3)]);
			}			
			//for (k = 0; k < NPR + 3; k++) dq2[k] = 0.;
			for (k = 0; k < NPR; k++){
				p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
				ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
			}
			//for (k = 0; k < NPR + 3; k++) dq1[k] = dq2[k] = dq3[k] = 0.;

			#if(STAGGERED)		
			if (is == 1){
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii + ref_1)*(1 + 2 * ref_3) + iz]) 
				    + 0.25*(double)(js)*dq2[B1] * ref_2 + 0.25*(double)(zs)*dq3[B1] * ref_3;
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + (ii + ref_1)*(1 + 2 * ref_3) + iz]) 
					+ 0.25*(double)(js)*dq2[B1] * ref_2 + 0.25*(double)(zs)*dq3[B1] * ref_3;
			}
			else{
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(js)*dq2[NPR] * ref_2 + 0.25*(double)(zs)*dq3[NPR] * ref_3);
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(js)*dq2[NPR] * ref_2 + 0.25*(double)(zs)*dq3[NPR] * ref_3);
			}
			if (zs == 1){
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz + ref_3)]) 
					+ 0.25*(double)(js)*dq2[B3] * ref_2 + 0.25*(double)(is)*dq1[B3] * ref_1;
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + (iz + ref_3)]) 
					+ 0.25*(double)(js)*dq2[B3] * ref_2 + 0.25*(double)(is)*dq1[B3] * ref_1;
			}
			else{
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(js)*dq2[NPR + 2] * ref_2 + 0.25*(double)(is)*dq1[NPR + 2] * ref_1);
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (receive_local[(NPR + 2)*(1 + 2 * ref_1)*(1 + 2 * ref_3)*(j2 - j1) + ij*(1 + 2 * ref_1)*(1 + 2 * ref_3) + ii*(1 + 2 * ref_3) + iz] + 0.25*(double)(js)*dq2[NPR + 2] * ref_2 + 0.25*(double)(is)*dq1[NPR + 2] * ref_1);
			}
#endif
		}
	}
}

__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3)
{
	int z, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR + 3], dq1[NPR + 3], dq2[NPR + 3], dq3[NPR + 3];
	double receive_local[(NPR + 3)*NG*(1 + 2 * REF_1)*(1 + 2 * REF_2)];
	int ii1, ij1, iz1;

	if (global_id < work_size){
//#pragma unroll NG
		for (z = z1; z < z2; z++){
			if (z1 < 0 && ref_3 == 1) iz = (NG - 1) + (z + 1) / (1 + ref_3);
			else if (ref_3 == 1) iz = (z - BS_3) / (1 + ref_3);
			else iz = z - z1;
			ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + ref_1)) / (1 + ref_1) + ref_1;
			ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + ref_2)) / (1 + ref_2) + ref_2;
			is = (((icurr - i1 - N1G) % (1 + ref_1) == 0) ? (-1) : (1));
			js = (((jcurr - j1 - N2G) % (1 + ref_2) == 0) ? (-1) : (1));
			if (z < 0){
				if (z == -3) zs = 1;
				else if (z == -2) zs = -1;
				else if (z == -1) zs = 1;
			}
			if (z>0){
				if (z == BS_3) zs = -1;
				else if (z == BS_3 + 1) zs = 1;
				else if (z == BS_3 + 2) zs = -1;
			}

			for (k = 0; k < NPR+3; k++){
				for (ii1 = ii - ref_1; ii1 <= ii + ref_1; ii1++)for (ij1 = ij - ref_2; ij1 <= ij + ref_2; ij1++)for (iz1 = 0; iz1 < z2-z1; iz1++){
					receive_local[k*(z2 - z1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + iz1*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii1 - (ii - ref_1))*(1 + 2 * ref_2) + (ij1 - (ij - ref_2))] = receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
					#if(PRESTEP2==100)
					//Store at full step the value at t in temp1
					if (nstep%timelevel_rec == timelevel_rec - 1|| nstep==-1){
						temp1receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] =  receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
					}
					//Store at prestep the time gradient in temp2
					if (nstep%timelevel_rec == timelevel - 1 || nstep%timelevel_rec == timelevel_rec - 1 || nstep==-1){
						temp1receive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] = (receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1]-temp1receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1])/(timelevel_rec*0.5*dt);
					}
					//Add gradient to boundary
					if (nstep%timelevel_rec != timelevel_rec - 1 && nstep != -1 && timelevel_rec > timelevel && icurr >= N1G && icurr<BS_1 + N1G && jcurr >= N2G && jcurr < BS_2 + N2G) {
						receive_local[k*(z2 - z1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + iz1*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii1 - (ii - ref_1))*(1 + 2 * ref_2) + (ij1 - (ij - ref_2))] = temp1receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] + (nstep + 1) % timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
					}					
					#elif(PRESTEP==-100)
					if(nstep%timelevel_rec==timelevel_rec-1 || nstep == -1){ 
						temp1receive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] = (receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1]-temp1receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1])/(timelevel_rec*0.5*dt);
					}
					if (nstep == -1 || timelevel_rec <= timelevel){
						temp1receive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] = 0.0;
						temp1receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] = receive[(k)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
					}

					if (nstep != -1 && timelevel_rec>timelevel && nstep%timelevel_rec!=timelevel_rec-1){
						temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] = receive[(k)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
						receive_local[k*(z2 - z1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + iz1*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii1 - (ii - ref_1))*(1 + 2 * ref_2) + (ij1 - (ij - ref_2))] = receive[k*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1] + (nstep + 1) % timelevel_rec*0.5*dt*timelevel*temp1receive[(k+NPR+3)*isize2*jsize2*(z2 - z1) + iz1*isize2*jsize2 + ii1*jsize2 + ij1];
					}
					#endif
				}
			}
			ii = ref_1;
			ij = ref_2;
			for (k = 0; k < NPR; k++){
				dq3[k] = 0.0;
				if (ref_3){
					if (iz == 0 || iz == NG - 1){
						avg[k] = 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						avg[k] += 0.125*(prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] + prim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1)*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2)*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						if (iz == 0){
							dq3[k] = slope_lim(avg[k], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)* (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij]);
						}
						else{
							dq3[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], avg[k]);
						}
					}
					else{
						dq3[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz+1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij]);
					}
				}
				dq1[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii - ref_1)*(1 + 2 * ref_2) + (ij)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii + ref_1)*(1 + 2 * ref_2) + (ij)]);
				dq2[k] = slope_lim(receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii)*(1 + 2 * ref_2) + (ij - ref_2)], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii)*(1 + 2 * ref_2) + (ij + ref_2)]);
			}

			for (k = 0; k < 3; k++){
				dq3[(NPR + k)] = 0.0;
				if (ref_3){
					if (iz == 0 || iz == NG - 1){
						avg[NPR + k] = 0.25*(psim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] +
							psim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2))*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						avg[NPR + k] += 0.25*(psim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1*(k == 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2*(k == 0))*(BS_3 + 2 * N3G) + (N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 1))] +
							psim[k*(ksize)+(icurr - (icurr - N1G) % (1 + ref_1) + ref_1*(k == 1))*isize + (jcurr - (jcurr - N2G) % (1 + ref_2) + ref_2*(k == 0))*(BS_3 + 2 * N3G) + (iz / (NG - 1) + N3G + (NG - 1 - iz) / (NG - 1)*(BS_3 - 2))]);
						if (iz == 0){
							dq3[NPR + k] = slope_lim(avg[NPR + k], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)* (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij]);
						}
						else{
							dq3[NPR + k] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], avg[NPR + k]);
						}
					}
					else{
						dq3[NPR + k] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz - 1) * (1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + (iz+1)*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij]);
					}
				}
				dq1[(NPR + k)] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii - ref_1)*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii + ref_1)*(1 + 2 * ref_2) + ij]);
				dq2[(NPR + k)] = slope_lim(receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii)*(1 + 2 * ref_2) + (ij - ref_2)], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij], receive_local[(NPR + k)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii)*(1 + 2 * ref_2) + (ij + ref_2)]);
			}
			//for (k = 0; k < NPR + 3; k++) dq1[k] = dq2[k] = dq3[k] = 0.;
			for (k = 0; k < NPR; k++){
				p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
				ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive_local[k*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(is)*dq1[k] * ref_1 + 0.25*(double)(js)*dq2[k] * ref_2 + 0.25*(double)(zs)*dq3[k] * ref_3;
			}

			#if(STAGGERED)
			if (is == 1){
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii + ref_1)*(1 + 2 * ref_2) + ij])
					+ 0.25*(double)(zs)*dq3[B1] * ref_3 + 0.25*(double)(js)*dq2[B1] * ref_2;
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + (ii + ref_1)*(1 + 2 * ref_2) + ij])
				    + 0.25*(double)(zs)*dq3[B1] * ref_3 + 0.25*(double)(js)*dq2[B1] * ref_2;
			}
			else{
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(zs)*dq3[NPR + 0] * ref_3 + 0.25*(double)(js)*dq2[NPR + 0] * ref_2);
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (receive_local[(NPR + 0)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(zs)*dq3[NPR + 0] * ref_3 + 0.25*(double)(js)*dq2[NPR + 0] * ref_2);
			}
			
			if (js == 1){
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + (ij + ref_2)])
					 + 0.25*(double)(zs)*dq3[B2] * ref_3 + 0.25*(double)(is)*dq1[B2] * ref_1;
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + (ij + ref_2)])
					 + 0.25*(double)(zs)*dq3[B2] * ref_3 + 0.25*(double)(is)*dq1[B2] * ref_1;
			}
			else{
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(zs)*dq3[NPR + 1] * ref_3 + 0.25*(double)(is)*dq1[NPR + 1] * ref_1);
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (receive_local[(NPR + 1)*(1 + 2 * ref_1)*(1 + 2 * ref_2)*(z2 - z1) + iz*(1 + 2 * ref_1)*(1 + 2 * ref_2) + ii*(1 + 2 * ref_2) + ij] + 0.25*(double)(zs)*dq3[NPR + 1] * ref_3 + 0.25*(double)(is)*dq1[NPR + 1] * ref_1);
			}
			#endif
		}
	}
}

__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
	}
}

__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
	}
}

__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
		}
	}
}

__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (k = 0; k < NPR; k++) {
					for (i = i1; i < i2; i++) {
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			else if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							+= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			else if (calc_corr == 6){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							= factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
					}
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							+= factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
					}
				}
			}

			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
					}
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]
							+= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]
							-= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			else if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							+= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
					}
				}
			}
			else if (calc_corr == 6 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
		}
	}
}

__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (k = 0; k < NPR; k++) {
					for (j = j1; j < j2; j++) {
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			else if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			else if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							+= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			else if (calc_corr == 6){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							= factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
					}
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							+= factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
					}
				}
			}

			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
					}
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
							+= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
							-= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							+= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
					}
				}
			}
			else if (calc_corr == 6 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
							= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
		}
	}
}

__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (k = 0; k < NPR; k++) {
					for (z = z1; z < z2; z++) {
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
					}
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
					}
				}
			}
			else if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
					}
				}
			}
			else if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
					}
				}
			}
			else if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							+= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
					}
				}
			}
			else if (calc_corr == 6){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;

					}
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							= factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
					}
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							+= factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
					}
				}
			}

			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
					}
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]
							+= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]
							-= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
			else if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							+= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
					}
				}
			}
			else if (calc_corr == 6 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]
							= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor; //times dt_old/dt_new to add in future code
					}
				}
			}
		}
	}
}

__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = 0.25*factor*(
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr] +
						pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += 0.25*factor*(
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr] +
						pv[k*(ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
				}
			}
		}
	}
}

__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						+ pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3] + pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						+ pv[k*(ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
				}
			}
		}
	}
}

__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + ref_2))*(1 + ref_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + ref_2))) / ((j2 - j1) / (1 + ref_2))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] = 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
						+ pv[k*(ksize)+(icurr + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] += 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
						+ pv[k*(ksize)+(icurr + ref_1)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G]);
				}
			}
		}
	}
}

__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int i;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1)) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1))) / ((z2 - z1)) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		else{
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
}

__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int j;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1)) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1))) / ((z2 - z1)) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		else{
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
}

__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int z;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1)) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1))) / ((j2 - j1)) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
			}
		}
		else{
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
			}
		}
	}
}

__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int i;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
		else{
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + ref_2)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
	}
}

__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int j;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + ref_3))*(1 + ref_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + ref_3))) / ((z2 - z1) / (1 + ref_3))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] = factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
		else{
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + ref_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + ref_1)*zsize2 + (zcurr - z1 - N3G) / (1 + ref_3)] += factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + ref_3]);
			}
		}
	}
}

__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3)
{
	int z;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + ref_2))*(1 + ref_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + ref_2))) / ((j2 - j1) / (1 + ref_2))*(1 + ref_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + ref_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] = factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G]);
			}
		}
		else{
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + ref_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + ref_1)*jsize2 + (jcurr - j1 - N2G) / (1 + ref_2)] += factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + z + N3G]);
			}
		}
	}
}

__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size)
{
	int i;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int z22 = z2 + D3;
	int zcurr = global_id % (z22 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z22 - z1)) / (z22 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (i = i1; i < i2; i++) {
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (i = i1; i < i2; i++) {
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
			else if (calc_corr == 2){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 3){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
			else if (calc_corr == 6){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = (receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = (receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += (receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += (receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (i = i1; i < i2; i++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (i = i1; i < i2; i++){
					if (jcurr >= j1 + N2G + d1*D2 && jcurr < j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
	}
}


__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size)
{
	int j;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int z22 = z2 + D3;
	int zcurr = global_id % (z22 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z22 - z1)) / (z22 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (j = j1; j < j2; j++) {
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (j = j1; j < j2; j++) {
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
			else if (calc_corr == 2){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 3){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= (temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= (temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
			}
			else if (calc_corr == 4){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G && zcurr < z2 + N3G && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					if (zcurr >= z1 + N3G && zcurr < z2 + N3G && icurr >= i1 + N1G && icurr < i2 + N1G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.;//0.5*(prim[3*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] - receive[1*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr-i1-N1G)*zsize2+(zcurr-z1-N3G)] / factor);
				}
			}
			else if (calc_corr == 5){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
			else if (calc_corr == 6){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = (receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = (receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += (receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += (receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (j = j1; j < j2; j++){
					if (zcurr >= z1 + N3G + e1*D3 && zcurr < z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
				for (j = j1; j < j2; j++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr < z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
	}
}

__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size)
{
	int z;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int j22 = j2 + 1;
	int jcurr = global_id % (j22 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j22 - j1)) / (j22 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				for (z = z1; z < z2; z++) {
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
				for (z = z1; z < z2; z++) {
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
			}
			else if (calc_corr == 2){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
			else if (calc_corr == 3){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= (temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= (temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
				}
			}
			else if (calc_corr == 5){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
			}
			else if (calc_corr == 6){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = (temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
				}
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = (receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = (receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += (receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += (receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				for (z = z1; z < z2; z++){
					if (jcurr >= j1 + N2G + e1*D2 && jcurr < j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr < i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
				for (z = z1; z < z2; z++){
					if (icurr >= i1 + N1G + d1*D1 && icurr < i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr < j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
		}
	}
}

__global__ void packsendE1corn(int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = global_id + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	}
}

__global__ void packsendE2corn(int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = global_id + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	}
}

__global__ void packsendE3corn(int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	}
}

__global__ void packsendE1corncourse(int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = global_id*(1 + ref_1) + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + ref_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	}
}

__global__ void packsendE2corncourse(int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_2){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = global_id*(1 + ref_2) + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + ref_2)*(BS_3 + 2 * N3G) + zcurr]);
	}
}

__global__ void packsendE3corncourse(int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_3){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id*(1 + ref_3) + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);
		else send[global_id] += 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + ref_3)]);
	}
}

__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = global_id + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[global_id] / factor;
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				temp1[global_id] = receive[global_id] - prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 2){
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
			}
			else if (calc_corr == 3){
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
			}
			else if (calc_corr == 5){
				temp1[global_id] += receive[global_id] - prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 6){
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (temp1[global_id]) / factor;
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				temp2[global_id] = factor*prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				temp2[global_id] += factor*prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				temp1[global_id] = (receive[global_id] - temp2[global_id]);
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
		}
	}
}

__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = global_id + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[global_id] / factor;
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				temp1[global_id] = receive[global_id] - prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 2){
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
			}
			else if (calc_corr == 3){
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
			}
			else if (calc_corr == 5){
				temp1[global_id] += receive[global_id] - prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 6){
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (temp1[global_id]) / factor;
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				temp2[global_id] = factor*prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				temp2[global_id] += factor*prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				temp1[global_id] = (receive[global_id] - temp2[global_id]);
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
		}
	}
}

__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (timelevel_rec <= timelevel){
			if (calc_corr == 1 && !PRESTEP && !PRESTEP2 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1) {
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[global_id] / factor;
			}
			else if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				temp1[global_id] = receive[global_id] - prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 2){
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
			}
			else if (calc_corr == 3){
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
			}
			else if (calc_corr == 5){
				temp1[global_id] += receive[global_id] - prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			else if (calc_corr == 6){
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = (temp1[global_id]) / factor;
			}
		}
		else{
			if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
				temp2[global_id] = factor*prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			else if ((calc_corr == 1 || calc_corr == 5)){
				temp2[global_id] += factor*prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
				temp1[global_id] = (receive[global_id] - temp2[global_id]);
			}
			if (calc_corr == 2 - !(PRESTEP || PRESTEP2) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
			}
			else if (calc_corr == 6 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
		}
	}
}
