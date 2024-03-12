
__device__  double slope_lim(double y1, double y2, double y3, int dir)
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

__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout)
{
	int i;
	double y[5], dq[5];
	double Dqm, Dqc, Dqp, aDqm, aDqp, aDqc, s, l, r, qa, qd, qe;

	y[0] = x1;
	y[1] = x2;
	y[2] = x3;
	y[3] = x4;
	y[4] = x5;

	/*CW1.7 */
	for (i = 1; i<4; i++) {
		Dqm = 2. *(y[i] - y[i - 1]);
		Dqp = 2. *(y[i + 1] - y[i]);
		Dqc = 0.5 *(y[i + 1] - y[i - 1]);
		aDqm = fabs(Dqm);
		aDqp = fabs(Dqp);
		aDqc = fabs(Dqc);
		s = Dqm*Dqp;
		Dqm = MY_MIN(aDqm, aDqp);
		if (aDqc< Dqm){
			if (Dqc>0.) dq[i] = (aDqc)*(double)(s>0.);
			else dq[i] = (-aDqc)*(double)(s>0.);
		}
		else{
			if (Dqc>0.) dq[i] = (Dqm)*(double)(s>0.);
			else dq[i] = (-Dqm)*(double)(s>0.);
		}

	}

	// CW1.6
	l = 0.5*(y[2] + y[1]) - (dq[2] - dq[1]) / 6.0;
	r = 0.5*(y[3] + y[2]) - (dq[3] - dq[2]) / 6.0;

	qa = (r - y[2])*(y[2] - l);
	qd = (r - l);
	qe = 6.0*(y[2] - 0.5*(l + r));

	if (qa <= 0.) {
		l = y[2];
		r = y[2];
	}

	if (qd*(qd - qe)<0.0) l = 3.0*y[2] - 2.0*r;
	else if (qd*(qd + qe)<0.0) r = 3.0*y[2] - 2.0*l;

	lout[0] = l;   //a_L,j
	rout[0] = r;
}

__device__ void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double *F) {
	double Sp;

	Sp = (x4 - x2) / (x5 - x1);
	F[0] = MY_MAX(0., MY_MIN(1., 10.*(Sp - 0.75)));
	if (fabs(x4 - x2) / MY_MIN(x4, x2) < 0.33) F[0] = 0;
}

__device__ double interp(double y1, double y2, double y3)
{
	double Dqm, Dqp, Dqc, s;
	/* woodward, or monotonized central, slope limiter */
	Dqm = (1.0)*(y2 - y1);
	Dqp = (1.0)*(y3 - y2);
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

//#include "decsCUDA.h"

__global__ void interpolate(double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + 2 * D3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3);
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3;
	jcurr += (N2G - 1)*D2;
	icurr += (N1G - 1)*D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zoffset = 0, z2 = 0, z3 = 0, z4 = 0;
	double x2, x3, x4;
	double temp;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int  zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (zdel) {
		if (zcurr == N3G - D3) {
			z2 = -1 * zdel;
			z3 = 0;
			z4 = 1 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G) {
			z2 = -zoffset - 1 * zdel;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G + zdel*zsize) {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
		else if (zcurr == BS_3 + N3G) {
			z2 = -1 * zdel*zsize;
			z3 = 0;
			z4 = 1 * zdel;
		}
		else if (zcurr - zoffset == BS_3 + N3G - zdel*zsize) {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + zdel*zsize;
		}
		else {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
	}

	if (k == 1) {
		#pragma unroll 9
		for (k = 0; k<NPR; k++) {
			x2 = p[MY_MAX(k*(ksize)+global_id + z2 - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x3 = p[k*(ksize)+global_id + z3];
			x4 = p[MY_MIN(k*(ksize)+global_id + z4 + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			temp = 0.5*interp(x2, x3, x4);
			dq1[k*(ksize)+global_id] = x3 - temp;
			dq2[k*(ksize)+global_id] = x3 + temp;
		}
	}
}

__global__ void reconstruct_internal(double* p, double* ps, const  double* __restrict__ dq1, const  double* __restrict__ dq2, const  double* __restrict__ gdet_GPU, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr*(BS_3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += D2;
	icurr += D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zoffset = 0, u;
	int zsize2 = 1, zoffset2 = 0;
	double temp[NPR];

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int  zlevel = 0, zlevel2 = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1){
		if (zoffset == 0){
			for (k = 0; k < NPR; k++) temp[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				for (k = 0; k < NPR; k++) temp[k] += p[k * (ksize)+global_id - zoffset + u] / ((double)zsize);
			}
			for (u = 0; u < zsize; u++){
				for (k = 0; k < NPR; k++) p[k*ksize + global_id - zoffset + u] = temp[k] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*(dq2[k*(ksize)+global_id - zoffset] - dq1[k*(ksize)+global_id - zoffset]);
			}

			temp[0] = 0.0;
			for (u = 0; u < zsize; u++) {
				temp[0] += ps[0 * (ksize)+global_id - zoffset + u] / ((double)zsize);
			}
			temp[2] = ps[2 * (ksize)+global_id - zoffset];
			for (u = 0; u < zsize; u++){
				ps[0 * ksize + global_id - zoffset + u] = temp[0] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*0.5*(dq2[B1*(ksize)+global_id - zoffset] + dq2[B1*(ksize)+global_id - isize - zoffset] - dq1[B1*(ksize)+global_id - zoffset] - dq1[B1*(ksize)+global_id - isize - zoffset]);
				ps[2 * ksize + global_id - zoffset + u] = temp[2] + ((double)u) / ((double)zsize)*(ps[2 * (ksize)+global_id - zoffset + zsize] - ps[2 * (ksize)+global_id - zoffset]);
			}
		}

		#if(N_LEVELS_1D_INT>0 && D3>0)
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - D2 - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize = (int)(0.001+pow(2.0, (double)zlevel));
		zoffset = (zcurr - N3G) % zsize;
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel2 = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel2 = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr + D2 - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize2 = (int)(0.001 + pow(2.0, (double)zlevel2));
		zoffset2 = (zcurr - N3G) % zsize2;
		#endif
		if (zoffset2 == 0) {
			if ((POLE_1 == 1 && jcurr - N2G < BS_2 / 2) && (jcurr != N2G)) {
				temp[1] = ps[1 * (ksize)+global_id - zoffset2];
				for (u = 0; u < zsize2; u++) {
					ps[1 * ksize + global_id - zoffset2 + u] = temp[1] + (((double)u + 0.5) - 0.5*(double)zsize2) / ((double)zsize)*0.5*(dq2[B2*(ksize)+global_id - (BS_3 + 2 * N3G) - zoffset] - dq1[B2*(ksize)+global_id - (BS_3 + 2 * N3G) - zoffset]);
					ps[1 * ksize + global_id - zoffset2 + u] += (((double)u + 0.5) - 0.5*(double)zsize2) / ((double)zsize2)*0.5*(dq2[B2*(ksize)+global_id - zoffset2] - dq1[B2*(ksize)+global_id - zoffset2]);
				}
			}
		}
		if (zoffset == 0) {
			if ((POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) && (jcurr + D2 != BS_2 + N2G)){
				temp[1] = ps[1 * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset];
				for (u = 0; u < zsize; u++){
					ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] = temp[1] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*0.5*(dq2[B2*(ksize)+global_id - zoffset] - dq1[B2*(ksize)+global_id - zoffset]);
					ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] +=  (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize2)*0.5*(dq2[B2*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset2] - dq1[B2*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset2]);
				}
			}
		}
	}
}