#include <cuda.h>
#include "config.h"

__device__ double slope_lim(double y1, double y2, double y3);

__device__ double slope_lim(double y1, double y2, double y3)
{
	double Dqm, Dqp, Dqc, s;
	/* woodward, or monotonized central, slope limiter */
	Dqm = (1.5)*(y2 - y1);
	Dqp = (1.5)*(y3 - y2);
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
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
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
			send[(NPR + 0)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+(i + N1G + (i1>N1G))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G +(i1>N1G))*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G), FACE1, 0)));
			send[(NPR + 1)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			send[(NPR + 2)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i+N1G)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
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
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
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
			send[(NPR + 0)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			send[(NPR + 1)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+icurr*isize + (j + N2G + (j1>N2G))*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G +(j1>N2G))];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G), icurr, FACE2, 0)));
			send[(NPR + 2)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
		}
		#endif
	}
}

__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (z = z1; z < z2; z++){
				send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
		}
		#if(STAGGERED)
		for (z = z1; z <z2; z++){
			send[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];
			send[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];
			send[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];
		}
		#endif
	}
}

__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;

	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (i = i1; i < i2; i += 1 + REF_1){
				send[k*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.125*(
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] +
					pv[k*(ksize)+(i + REF_1 + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
		#if(STAGGERED)
		for (i = i1; i <i2; i += 1 + REF_1){
			send[(NPR + 0)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
			ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
			ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr + REF_2)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
			ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr + REF_2)]);// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))));

			send[(NPR + 1)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0))) +
			ps[1 * (ksize)+(i + N1G)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0))) +
			ps[1 * (ksize)+(i + N1G + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE2, 0))) +
			ps[1 * (ksize)+(i + N1G + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]);// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE2, 0))));

			send[(NPR + 2)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0))) +
			ps[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G, FACE3, 0))) +
			ps[2 * (ksize)+(i + N1G + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE3, 0))) +
			ps[2 * (ksize)+(i + N1G + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]);// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + REF_1, FACE3, 0))));
		}
		#endif
	}
}

__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (j = j1; j < j2; j += 1 + REF_2){
				send[k*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.125*(
					pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] +
					pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
		#if(STAGGERED)
		for (j = j1; j <j2; j += 1 + REF_2){
			send[(NPR + 0)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)]);//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE1, 0))));

			send[(NPR + 1)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr, FACE2, 0))) +
			ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr, FACE2, 0))) +
			ps[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr + REF_1, FACE2, 0))) +
			ps[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))]);//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr + REF_1, FACE2, 0))));

			send[(NPR + 2)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
				0.25*(ps[2 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0))) +
			ps[2 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE3, 0))) +
			ps[2 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (j + N2G)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr + REF_1, FACE3, 0))) +
			ps[2 * (ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)]);// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr + REF_1, FACE3, 0))));
		}
		#endif
	}
}

__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		for (k = 0; k < NPR; k++){
			//#pragma unroll NG
			for (z = z1; z < z2; z += 1 + REF_3){
				send[k*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = 0.125*(
					pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + REF_3] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] +
					pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + REF_3] +
					pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G + REF_3]);
			}
		}
		#if(STAGGERED)
		for (z = z1; z <z2; z += 1 + REF_3){
			send[(NPR + 0)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
				0.25*(ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE1, 0))) +
			ps[0 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]);//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE1, 0))));

			send[(NPR + 1)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
				0.25*(ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) +
			ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) +
			ps[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE2, 0))) +
			ps[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]);//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE2, 0))));

			send[(NPR + 2)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
				0.25*(ps[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]+//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0))) +
			ps[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE3, 0))) +
			ps[2 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr)]+// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE3, 0))) +
			ps[2 * (ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr + REF_2)]);// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr + REF_1, FACE3, 0))));
		}
		#endif
	}
}

__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		//When at timestep where n_rec does not evolve
		if (nstep != -1 && timelevel_rec > timelevel){
			for (k = 0; k < NPR; k++){
				for (i = i1; i < i2; i++){
					p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
			}
			for (i = i1; i < i2; i++){
				/*ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				if((jcurr<N2G || jcurr>=BS_2+N2G || zcurr<N3G || zcurr>=BS_3+N3G)&& update_staggered==1){
				ps[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				}
				else psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				*/
			}
		}

		//When at same timestep where n_rec evolves
		if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec > timelevel){
			for (k = 0; k < NPR; k++){
				for (i = i1; i < i2; i++){
					p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
				for (i = i1; i < i2; i++){
					tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
						= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] - p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				}
			}
			for (i = i1; i < i2; i++){
				/*ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				if((jcurr<N2G || jcurr>=BS_2+N2G || zcurr<N3G || zcurr>=BS_3+N3G)&& update_staggered==1){
				ps[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				}
				else psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				*/
			}
			for (i = i1; i < i2; i++){
				tempreceive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1 < 0))*(BS_2 + 2 * N2G) + (jcurr)]//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0))) 
				-ps[0 * (ksize)+(i + N1G + (i1 < 0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				tempreceive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)))
				-ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				tempreceive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)))
				-ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
			}
		}

		//Reset gradient to 0 at eg refinement steps
		if (nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				//#pragma unroll NG
				for (i = i1; i < i2; i++){
					tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] = 0.0;
				}
			}
		}

		//Reset primitve variables
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (i = i1; i < i2; i++){
					p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
			}
			#if(STAGGERED)
			for (i = i1; i <i2; i++){
				if ((jcurr<N2G || jcurr >= BS_2 + N2G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
					ps[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1<0))*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0)));
					psh[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1<0))*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0)));
				}
				else psh[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE1, 0)));
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));

				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
			}
			#endif
		}
	}
}

__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;
	if (global_id < work_size){
		if (reverse == 0){
			//When at timestep where n_rec does not evolve
			if (nstep != -1 && timelevel_rec > timelevel){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
						ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					}
				}
				for (j = j1; j < j2; j++){
					/*ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					if((icurr<N1G || icurr>=BS_1+N1G || zcurr<N3G || zcurr>=BS_3+N3G) && update_staggered==1 ){
					ps[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					}
					else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					*/
				}
			}

			//When at same timestep where n_rec evolves
			if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec > timelevel){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
						ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					}
					for (j = j1; j < j2; j++){
						tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
							= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] - p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
					}
				}
				for (j = j1; j < j2; j++){
					/*ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					if((icurr<N1G || icurr>=BS_1+N1G || zcurr<N3G || zcurr>=BS_3+N3G) && update_staggered==1 ){
					ps[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					}
					else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
					*/
				}
				for (j = j1; j < j2; j++){
					tempreceive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
						= (receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) 
					-ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
					tempreceive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
						= (receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1 < 0))] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0))) 
					-ps[1 * (ksize)+icurr*isize + (j + N2G + (j1 < 0))*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
					tempreceive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
						= (receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0))) 
					-ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				}
			}

			//Reset gradient to 0 at eg refinement steps
			if (nstep == -1){
				for (k = 0; k < NPR + 3; k++){
					//#pragma unroll NG
					for (j = j1; j < j2; j++){
						tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] = 0.0;
					}
				}
			}

			//Reset primitve variables
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				for (k = 0; k < NPR; k++){
					//#pragma unroll NG
					for (j = j1; j < j2; j++){
						p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
						ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					}
				}
				#if(STAGGERED)
				for (j = j1; j <j2; j++){
					ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
					if ((icurr<N1G || icurr >= BS_1 + N1G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
						ps[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
							gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1<0))]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0)));
						psh[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
							gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1<0))]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0)));
					}
					else psh[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0)));
					ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
					psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));

					psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
				}
				#endif
			}
		}
		else{
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (j = j1; j < j2; j++){
					if (k == 3 || k == 4 || k == 6 || k == 7) factor = -1.;
					else factor = 1.;
					p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
			}
			#if(STAGGERED)
			for (j = j1; j <j2; j++){
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
				if (update_staggered == 1){
					//ps[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
					//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
					//psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
					//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
				}
				//else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
				//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			}
			#endif
		}
	}
}

__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	
	if (global_id < work_size){
		//When at timestep where n_rec does not evolve
		if (nstep != -1 && timelevel_rec > timelevel){
			for (k = 0; k < NPR; k++){
				for (z = z1; z < z2; z++){
					p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
					ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				}
			}
			for (z = z1; z < z2; z++){
				/*ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				if((icurr<N1G || icurr>=BS_1+N1G || jcurr<N2G || jcurr>=BS_2+N2G) && update_staggered==1 ){
				ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				}
				else psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				*/
			}
		}

		//When at same timestep where n_rec evolves
		if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec > timelevel){
			for (k = 0; k < NPR; k++){
				for (z = z1; z < z2; z++){
					p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
					ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				}
				for (z = z1; z < z2; z++){
					tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] - p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
				}
			}
			for (z = z1; z < z2; z++){
				/*ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				if((icurr<N1G || icurr>=BS_1+N1G || jcurr<N2G || jcurr>=BS_2+N2G) && update_staggered==1 ){
				ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				}
				else psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
				*/
			}
			for (z = z1; z < z2; z++){
				tempreceive[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
					= (receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) 
				-ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
				tempreceive[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
					= (receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) 
				-ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
				tempreceive[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
					= (receive[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)] //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0))) 
				-ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1 < 0))]) / (0.5*dt*timelevel_rec);
			}
		}

		//Reset gradient to 0 at eg refinement steps
		if (nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				//#pragma unroll NG
				for (z = z1; z < z2; z++){
					tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] = 0.0;
				}
			}
		}

		//Reset primitve variables
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR; k++){
				//#pragma unroll NG
				for (z = z1; z < z2; z++){
					p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
					ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				}
			}
			#if(STAGGERED)
			for (z = z1; z <z2; z++){
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
				if ((icurr<N1G || icurr >= BS_1 + N1G || jcurr<N2G || jcurr >= BS_2 + N2G) && update_staggered == 1){
					ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
						gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
					psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
						gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
				}
				else psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			}
			#endif
		}
	}
}
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int i, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;
	
	if (global_id < work_size){
		//#pragma unroll NG
		for (i = i1; i < i2; i++){
			if (i1 < 0 && REF_1 == 1) ii = REF_1;
			else if (REF_1 == 1) ii = 0;
			else ii = i - i1;
			ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + REF_2)) / (1 + REF_2) + REF_2;
			iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + REF_3)) / (1 + REF_3) + REF_3;

			is = ((i == i1) ? (-1) : (1));
			js = (((jcurr - j1 - N2G) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = (((zcurr - z1 - N3G) % (1 + REF_3) == 0) ? (-1) : (1));
			for (k = 0; k < NPR; k++){
				if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
					temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];
					temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];
				}

				if (nstep == -1 && timelevel_rec > timelevel)temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];

				if (nstep != -1 && timelevel_rec > timelevel){
					receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] +
						(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] - temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz]);
				}
			}
			for (k = 0; k < NPR; k++){
				avg = 0.5*(prim[k*(ksize)+(N1G + (1 - ii)*(BS_1 - 1))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + prim[k*(ksize)+(ii + N1G + (1 - ii)*(BS_1 - 2))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
				if (ii == 0){
					dq1 = slope_lim(avg, receive[k*jsize2*zsize2*(i2 - i1) + 0 * jsize2*zsize2 + ij*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + REF_1*jsize2*zsize2 + ij*zsize2 + iz]);
				}
				else{
					dq1 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + 0 * jsize2*zsize2 + ij*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + REF_1*jsize2*zsize2 + ij*zsize2 + iz], avg);
				}
				//if(jcurr-N2G-j1<(j2-j1)/2)dq2=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij+REF_2)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
				//else dq2=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij-REF_2)*zsize2+iz];
				//if(zcurr-N3G-z1<(z2-z1)/2)dq3=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz+REF_3]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
				//else dq3=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz-REF_3];
				dq2 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij - REF_2)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]);
				dq3 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + (iz - REF_3)], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + (iz + REF_3)]);
				//dq2=dq3=0.;
				p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}

			#if(STAGGERED)
			if (js == 1){
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]) /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]) /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			}
			else{
				ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
				psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			}
			if (zs == 1){
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + (iz + REF_3)]) /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + (iz + REF_3)]) /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
			}
			else{
				ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
				psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
			}


			//ps[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			#endif
		}
	}
}

__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int j, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;

	if (global_id < work_size){
//#pragma unroll NG
		for (j = j1; j < j2; j++){
			if (j1 < 0 && REF_2 == 1) ij = REF_2;
			else if (REF_2 == 1) ij = 0;
			else ij = j - j1;

			ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + REF_1)) / (1 + REF_1) + REF_1;
			iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + REF_3)) / (1 + REF_3) + REF_3;

			is = (((icurr - i1 - N1G) % (1 + REF_1) == 0) ? (-1) : (1));
			js = ((j == j1) ? (-1) : (1));
			zs = (((zcurr - z1 - N3G) % (1 + REF_3) == 0) ? (-1) : (1));

			for (k = 0; k < NPR; k++){
				if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
					temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];
					temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];
				}

				if (nstep == -1 && timelevel_rec > timelevel)temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];

				if (nstep != -1 && timelevel_rec > timelevel){
					receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] +
						(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] - temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz]);
				}
			}
			for (k = 0; k < NPR; k++){
				avg = 0.5*(prim[k*(ksize)+icurr*isize + (N2G + (1 - ij)*(BS_2 - 1))*(BS_3 + 2 * N3G) + zcurr] + prim[k*(ksize)+icurr*isize + (ij + N2G + (1 - ij)*(BS_2 - 2))*(BS_3 + 2 * N3G) + zcurr]);
				if (ij == 0){
					dq2 = slope_lim(avg, receive[k*isize2*zsize2*(j2 - j1) + 0 * isize2*zsize2 + ii*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + REF_2*isize2*zsize2 + ii*zsize2 + iz]);
				}
				else{
					dq2 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + 0 * isize2*zsize2 + ii*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + REF_2*isize2*zsize2 + ii*zsize2 + iz], avg);
				}
				//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii+REF_1)*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
				//else dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii-REF_1)*zsize2+iz];
				//if(zcurr-N3G-z1<(z2-z1)/2)dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz+REF_3)]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
				//else dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz-REF_3)];
				//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii+REF_1)*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
				//else dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii-REF_1)*zsize2+iz];
				//dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz+REF_3)]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
				dq1 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii - REF_1)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]);
				dq3 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + (iz - REF_3)], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + (iz + REF_3)]);
				//dq1=dq3=0.;
				p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}

#if(STAGGERED)
			if (is == 1){
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]) /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]) /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			}
			else{
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			}
			if (zs == 1){
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + (iz + REF_3)]) /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + (iz + REF_3)]) /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			}
			else{
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			}

			//ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//ps[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
#endif
		}
	}
}

__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double * p, double * ph, double * ps, double * psh, double * prim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size)
{
	int z, ii, ij, iz, is, js, zs, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;

	if (global_id < work_size){
//#pragma unroll NG
		for (z = z1; z < z2; z++){
			if (z1 < 0 && REF_3 == 1) iz = REF_3;
			else if (REF_3 == 1) iz = 0;
			else iz = z - z1;
			ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + REF_1)) / (1 + REF_1) + REF_1;
			ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + REF_2)) / (1 + REF_2) + REF_2;
			is = (((icurr - i1 - N1G) % (1 + REF_1) == 0) ? (-1) : (1));
			js = (((jcurr - j1 - N2G) % (1 + REF_2) == 0) ? (-1) : (1));
			zs = ((z == z1) ? (-1) : (1));

			for (k = 0; k < NPR; k++){
				if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
					temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];
					temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];
				}

				if (nstep == -1 && timelevel_rec > timelevel)temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];

				if (nstep != -1 && timelevel_rec > timelevel){
					receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] +
						(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] - temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij]);
				}
			}
			for (k = 0; k < NPR; k++){
				avg = 0.5*(prim[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (N3G + (1 - iz)*(BS_3 - 1))] + prim[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (iz + N3G + (1 - iz)*(BS_3 - 1))]);
				if (iz == 0){
					dq3 = slope_lim(avg, receive[k*isize2*jsize2*(z2 - z1) + 0 * isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + REF_3*isize2*jsize2 + ii*jsize2 + ij]);
				}
				else{
					dq3 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + 0 * isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + REF_3*isize2*jsize2 + ii*jsize2 + ij], avg);
				}
				//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + (ii+REF_1)*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
				//else dq1=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + (ii-REF_1)*jsize2+ij];
				//if(jcurr-N2G-j1<(j2-j1)/2)dq2=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+(ij+REF_2)]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
				//else dq2=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+(ij-REF_2)];
				dq1 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii - REF_1)*jsize2 + ij]);
				dq2 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii)*jsize2 + (ij + REF_2)], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii)*jsize2 + (ij - REF_2)]);
				//dq1=dq2=0.;
				p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
				ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			}

#if(STAGGERED)
			if (is == 1){
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij]) /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij]) /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			}
			else{
				ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
				psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			}
			if (js == 1){
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + (ij + REF_2)]) /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + (ij + REF_2)]) /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			}
			else{
				ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
				psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			}
			//ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+0)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+0)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+1)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+1)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+2)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+2)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (i = i1; i < i2; i++){
						temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
							+= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
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
			else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
					}
				}
			}
			if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
					}
				}
			}
			if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (j = j1; j < j2; j++){
						temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
							+= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
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
			else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
					}
				}
			}
			if (calc_corr == 2){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
					}
				}
			}
			if (calc_corr == 3){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
					}
				}
			}
			if (calc_corr == 5){
				for (k = 0; k < NPR; k++){
					for (z = z1; z < z2; z++){
						temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
							+= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
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
			else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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
		}
	}
}

__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int i, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*factor*(
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
						pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (i = i1; i < i2; i++){
					send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += 0.25*factor*(
						pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
						pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				}
			}
		}
	}
}

__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						+ pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (j = j1; j < j2; j++){
					send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						+ pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				}
			}
		}
	}
}

__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int z, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
						+ pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
				}
			}
		}
		else{
			for (k = 0; k < NPR; k++){
//#pragma unroll NG
				for (z = z1; z < z2; z++){
					send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += 0.25*factor*(
						pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
						+ pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
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

__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int i;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
		else{
			//k=2
			for (i = i1; i < i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3
			for (i = i1; i < i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
}

__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int j;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
		else{
			//k=1;
			for (j = j1; j < j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
			}
			//k=3;
			for (j = j1; j < j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
}

__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size)
{
	int z;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (global_id < work_size){
		if (first_timestep == 1){
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
			}
		}
		else{
			//k=1;
			for (z = z1; z < z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
			}
			//k=2;
			for (z = z1; z < z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
					if (zcurr >= z1 + N3G && zcurr < z2 + N3G && icurr >= i1 + N1G && icurr < i2 + N1G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.;//0.5*(prim[3*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] - receive[1*isize2*zsize2*(j2-j1) + (j - j1)*isize2*zsize2 + (icurr-i1-N1G)*zsize2+(zcurr-z1-N3G)] / factor);
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
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

__global__ void packsendE1corncourse(int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = global_id*(1 + REF_1) + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
		//double avg=0.5*(pv[1*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr+REF_1)*isize+jcurr*(BS_3+2*N3G)+zcurr]);
		//pv[1*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
		//pv[1*(ksize)+(icurr+REF_1)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
	}
}

__global__ void packsendE2corncourse(int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = global_id*(1 + REF_2) + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
		else send[global_id] += 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
		//double avg=0.5*(pv[2*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]);
		//pv[2*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
		//pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=avg;
	}
}

__global__ void packsendE3corncourse(int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size){
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id*(1 + REF_3) + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id < work_size){
		if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)]);
		else send[global_id] += 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)]);
		//double avg=0.5*(pv[3*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		//pv[3*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
		//pv[3*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
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
			if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
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
			else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * PRESTEP * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
				prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
			}
			else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
				temp1[global_id] += (receive[global_id] - temp2[global_id]);
			}
		}
	}
}
