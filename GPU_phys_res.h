#if(RESISTIVE)
__device__ void primtoflux_res(double* pr, struct of_state_res* q_res, int dir, struct of_geom* geom, double* flux)
{
	#if(RESISTIVE)
	int k;
	double alpha, beta[NDIM], Ecov[3], Bcov[3];
	double sqrtgamma_inv;

	/* particle number flux */
	flux[RHO] = pr[RHO] * q_res->ucon[dir];

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc_res(pr, dir, geom, q_res, &flux[UU]);
	flux[UU] += flux[RHO];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-(geom->gcon[0]));

	//Beta in 3+1
	beta[0] = 0;
	beta[1] = geom->gcon[1] * alpha * alpha;
	beta[2] = geom->gcon[2] * alpha * alpha;
	beta[3] = geom->gcon[3] * alpha * alpha;

	//Inverse of 3-metric
	sqrtgamma_inv = alpha / (geom->g);

	/*Maxwell tensor */
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	if (dir == 0) {
		flux[E1] = pr[E1];
		flux[E2] = pr[E2];
		flux[E3] = pr[E3];
	}
	else {
		flux[E1] = beta[1] * pr[E1 + (dir - 1)] - beta[dir] * pr[E1];
		flux[E2] = beta[2] * pr[E1 + (dir - 1)] - beta[dir] * pr[E2];
		flux[E3] = beta[3] * pr[E1 + (dir - 1)] - beta[dir] * pr[E3];
		flux[E1] -= lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 0 - (dir - 1))]);
		flux[E2] -= lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 1 - (dir - 1))]);
		flux[E3] -= lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 2 - (dir - 1))]);
	}

	/* dual of Maxwell tensor */
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	if (dir == 0) {
		flux[B1] = pr[B1];
		flux[B2] = pr[B2];
		flux[B3] = pr[B3];
	}
	else {
		flux[B1] = beta[1] * pr[B1 + (dir - 1)] - beta[dir] * pr[B1];
		flux[B2] = beta[2] * pr[B1 + (dir - 1)] - beta[dir] * pr[B2];
		flux[B3] = beta[3] * pr[B1 + (dir - 1)] - beta[dir] * pr[B3];
		flux[B1] += lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 0 - (dir - 1))]);
		flux[B2] += lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 1 - (dir - 1))]);
		flux[B3] += lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 2 - (dir - 1))]);
	}

	//Entropy advection
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * (GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA);
	#endif

	PLOOP flux[k] *= geom->g;
	#endif
}

/* calculate magnetic field four-vector */
__device__ void econ_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* econ)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], E_dot_v, Bcov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[1] * alpha;
	ncon[2] = -geom->gcon[2] * alpha;
	ncon[3] = -geom->gcon[3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA!
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	E_dot_v = alpha * (pr[E1] * ucov[1] + pr[E2] * ucov[2] + pr[E3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	econ[0] = (E_dot_v)*ncon[0];
	econ[1] = (E_dot_v)*ncon[1] + gamma * (alpha * pr[E1]) + (alpha * alpha / geom->g) * (ucov[2] * Bcov[2] - ucov[3] * Bcov[1]);
	econ[2] = (E_dot_v)*ncon[2] + gamma * (alpha * pr[E2]) + (alpha * alpha / geom->g) * (ucov[3] * Bcov[0] - ucov[1] * Bcov[2]);
	econ[3] = (E_dot_v)*ncon[3] + gamma * (alpha * pr[E3]) + (alpha * alpha / geom->g) * (ucov[1] * Bcov[1] - ucov[2] * Bcov[0]);

	return;
	#endif
}

/* calculate magnetic field four-vector */
__device__ void bcon_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* bcon)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], B_dot_v, Ecov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[1] * alpha;
	ncon[2] = -geom->gcon[2] * alpha;
	ncon[3] = -geom->gcon[3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	B_dot_v = alpha * (pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	bcon[0] = (B_dot_v)*ncon[0];
	bcon[1] = (B_dot_v)*ncon[1] + gamma * (alpha * pr[B1]) - (alpha * alpha / geom->g) * (ucov[2] * Ecov[2] - ucov[3] * Ecov[1]);
	bcon[2] = (B_dot_v)*ncon[2] + gamma * (alpha * pr[B2]) - (alpha * alpha / geom->g) * (ucov[3] * Ecov[0] - ucov[1] * Ecov[2]);
	bcon[3] = (B_dot_v)*ncon[3] + gamma * (alpha * pr[B3]) - (alpha * alpha / geom->g) * (ucov[1] * Ecov[1] - ucov[2] * Ecov[0]);

	return;
	#endif
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc_res(double* pr, int dir, struct of_geom* geom, struct of_state_res* q_res, double* mhd)
{
	#if(RESISTIVE)
	int j, lambda, beta, kappa;
	double P, w, bsq, esq, eta, ptot, mhd_u[NDIM], mhd_d[NDIM];

	//Calculate contraction term
	DLOOPA{
		mhd_u[j] = 0.;
		for (lambda = 0; lambda < NDIM; lambda++)for (beta = 0; beta < NDIM; beta++)for (kappa = 0; kappa < NDIM; kappa++) {
			mhd_u[j] += q_res->ucov[lambda] * q_res->ecov[beta] * q_res->bcov[kappa] * (q_res->ucon[dir] * (1.0 / geom->g) * lvc4u(j, lambda, beta, kappa) + q_res->ucon[j] * (1.0 / geom->g) * lvc4u(dir, lambda, beta, kappa));
		}
	}
	lower(mhd_u, geom->gcov, mhd_d);

	#if DOHELM   
	eos_mode_rhou_pres(pr[RHO], pr[UU], &P); // Helmholtz EOS
	#else
	P = (GAMMA - 1.) * pr[UU]; // Ideal gas EOS
	#endif

	w = P + pr[RHO] + pr[UU];
	bsq = dot(q_res->bcon, q_res->bcov);
	esq = dot(q_res->econ, q_res->ecov);
	eta = w + (bsq + esq);
	ptot = P + 0.5 * (bsq + esq);

	//single row of mhd stress tensor, first index up, second index down
	DLOOPA mhd[j] = eta * q_res->ucon[dir] * q_res->ucov[j] + ptot * delta(dir, j) - q_res->bcon[dir] * q_res->bcov[j] - q_res->econ[dir] * q_res->ecov[j] + mhd_d[j];
	#endif
}

/* add in geometrical and cooling source terms to equations of motion */
__device__ void source_res(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double* q, double Dt, const  double* __restrict__ conn_GPU, struct of_state_res* q_res, double r) {
	double conn, mhd_res[NDIM][NDIM];
	int k;
	double alpha, beta[4], gamma;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr * (BS_2 + 2 * N2G) + jcurr;
	#endif

	mhd_calc_res(ph, 0, geom, q_res, mhd_res[0]);
	mhd_calc_res(ph, 1, geom, q_res, mhd_res[1]);
	mhd_calc_res(ph, 2, geom, q_res, mhd_res[2]);
	mhd_calc_res(ph, 3, geom, q_res, mhd_res[3]);

	/* contract mhd stress tensor with connection */
	PLOOP dU[k] = 0.;

	#pragma unroll 4
	for (k = 0; k < NDIM; k++) {
		#if(NSY)
		dU[UU] += mhd_res[0][k] * conn_GPU[0 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[1][k] * conn_GPU[4 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[2][k] * conn_GPU[7 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd_res[3][k] * conn_GPU[9 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[1][k] * conn;
		dU[U1] += mhd_res[0][k] * conn;
		conn = conn_GPU[2 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[0][k] * conn;
		conn = conn_GPU[3 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[0][k] * conn;
		conn = conn_GPU[5 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[1][k] * conn;
		conn = conn_GPU[6 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[1][k] * conn;
		conn = conn_GPU[8 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[2][k] * conn;
		#else
		dU[UU] += mhd_res[0][k] * conn_GPU[0 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[1][k] * conn_GPU[4 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[2][k] * conn_GPU[7 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd_res[3][k] * conn_GPU[9 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[1][k] * conn;
		dU[U1] += mhd_res[0][k] * conn;
		conn = conn_GPU[2 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[0][k] * conn;
		conn = conn_GPU[3 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[0][k] * conn;
		conn = conn_GPU[5 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[1][k] * conn;
		conn = conn_GPU[6 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[1][k] * conn;
		conn = conn_GPU[8 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[2][k] * conn;
		#endif
	}

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//Beta in 3+1
	beta[1] = geom->gcon[1] * alpha * alpha;
	beta[2] = geom->gcon[2] * alpha * alpha;
	beta[3] = geom->gcon[3] * alpha * alpha;

	//Calculate relative Lorentz factor
	gamma = q_res->ucon[0] * alpha;

	//Calculate explicit part of electric current J sourceterm
	dU[E1] = -alpha * q[0] * ph[U1] / gamma + beta[1] * q[0];
	dU[E2] = -alpha * q[0] * ph[U2] / gamma + beta[2] * q[0];
	dU[E3] = -alpha * q[0] * ph[U3] / gamma + beta[3] * q[0];

	//Add cooling term if needed
	#if (COOL_DISK)
	//misc_source(ph, icurr, jcurr, geom, q, dU, r, Dt);
	#endif

	PLOOP dU[k] *= geom->g;
}

//find ucon, ucov, bcon, bcov from primitive variables */
__device__ void get_state_res(double* pr, struct of_geom* geom, struct of_state_res* q_res
	#if(CALC_MDOT)
	, double magnetic_density_scale
	#endif
)
{
	#if(RESISTIVE)
	//get ucon
	ucon_calc(pr, geom, q_res->ucon);
	lower(q_res->ucon, geom->gcov, q_res->ucov);

	//get bcon
	bcon_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->bcon);
	lower(q_res->bcon, geom->gcov, q_res->bcov);

	#if(CALC_MDOT)
	int k;
	for (k = 0; k < NDIM; k++) {
		q_res->bcon[k] *= magnetic_density_scale;
		q_res->bcov[k] *= magnetic_density_scale;
	}
	#endif

	//get econ
	econ_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->econ);
	lower(q_res->econ, geom->gcov, q_res->ecov);
	#endif
}

//Calculate wavespeed assuming it is c
__device__ void vchar_res(struct of_geom* geom, int js, double* vmax, double* vmin) {
	double sqrtgamma, ncon_js, alpha, beta, vm, vp;
	alpha = 1. / sqrt(-geom->gcon[0]);
	beta = geom->gcon[js] * alpha * alpha;
	ncon_js = -alpha * geom->gcon[js];

	if (js == 1) sqrtgamma = sqrt(geom->gcon[4] + ncon_js * ncon_js);
	else if (js == 2) sqrtgamma = sqrt(geom->gcon[7] + ncon_js * ncon_js);
	else sqrtgamma = sqrt(geom->gcon[9] + ncon_js * ncon_js);

	vp = alpha * sqrtgamma - beta;
	vm = -alpha * sqrtgamma - beta;

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
	}
	else {
		*vmax = vm;
		*vmin = vp;
	}
}

//Calculate wavespeed in ideal limit
__device__ void vchar_res2(double* pr, struct of_state_res* q, struct of_geom* geom, int dir, double* vmax, double* vmin)
{
	double discr, vp, vm, va2, cs2, cms2;
	double Acon_0, Acon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	if (dir == 1) {
		Acon_0 = geom->gcon[1];
		Acon_js = geom->gcon[4];
	}
	else if (dir == 2) {
		Acon_0 = geom->gcon[2];
		Acon_js = geom->gcon[7];
	}
	else if (dir == 3) {
		Acon_0 = geom->gcon[3];
		Acon_js = geom->gcon[9];
	}

	double w, bsq, eta;
	// EOS-specific calls:
	#if (0)//(DOHELM) DIMARK: not working yet!	// 1. Helmholtz EOS
	double cs2_helm;
	eos_mode_rhou_pres_cs2(gpu_eos_table, pr[RHO], pr[UU], &P, &cs2_helm);
	w = pr[RHO] + pr[UU] + P;
	#else
		// 2. Ideal gas EOS
		#if(AMD)
		w = fma(GAMMA, pr[UU], pr[RHO]);
		#else
		w = pr[RHO] + GAMMA * pr[UU];
		#endif
	#endif
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;

	/* find fast magnetosonic speed */
	// EOS-specific calls:
	#if (0) //(DOHELM) DIMARK: not working yet!
	// 1. Helmholtz EOS
	// cs2 was already calculated above
	cs2 = cs2_helm;
	#else
	// 2. Ideal gas EOS
	cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / w;
	#endif

	va2 = bsq / eta;
	cms2 = cs2 + va2 - cs2 * va2;	/* and there it is... */

	//check on it!
	if (cms2 < 0.) cms2 = SMALL;
	if (cms2 > 1.) cms2 = 1.;

	//now require that speed of wave measured by observer q->ucon is cms2
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	AB = Acon_0;
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;
	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2. * fma(-(AB + AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4. * A * C);
	#else
	A = Bu2 - (Bsq + Bu2) * cms2;
	B = 2. * (AuBu - (AB + AuBu) * cms2);
	C = Au2 - (Asq + Au2) * cms2;
	discr = B * B - 4. * A * C;
	#endif

	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.;
	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	*vmax = MY_MAX(vp, vm);
	*vmin = MY_MIN(vp, vm);

	return;
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower_3(double* ucon, double gcov[10], double* ucov)
{
	#if(RESISTIVE)
	ucov[0] = gcov[4] * ucon[0] + gcov[5] * ucon[1] + gcov[6] * ucon[2];
	ucov[1] = gcov[5] * ucon[0] + gcov[7] * ucon[1] + gcov[8] * ucon[2];
	ucov[2] = gcov[6] * ucon[0] + gcov[8] * ucon[1] + gcov[9] * ucon[2];
	return;
	#endif
}

//4D Levi-cevita symbol (not tensor)
__device__ double lvc4u(int i, int j, int k, int l) {
	double lvc4u;

	if ((i == j) || (i == k) || (i == l) || (j == k) || (j == l) || (k == l)) {
		lvc4u = 0.0;
	}
	else if ((i + j == 1) || (i + j == 5)) {
		if ((j + k) % 4 == 3) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if ((i + j == 2) || (i + j == 4)) {
		if ((j + k) % 4 == 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if (i + j == 3) {
		if ((j + k) % 4 != 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else lvc4u = 0.0;

	return (lvc4u);
}

//3D Levi-cevita symbol (not tensor)
__device__ double lvc3u(int i, int j, int k) {
	double lvc3u;

	if ((i == j) || (j == k) || (k == i)) lvc3u = 0.;
	else if ((i + 1 == j) || (i - 2 == j)) lvc3u = 1.;
	else lvc3u = -1.;

	return (lvc3u);
}

__device__ double divE_calc(double* p, const  double* __restrict__ gdet, double _dx1, double _dx2, double _dx3, int ii, int jj, int zz) {
	#if(RESISTIVE)
	double divE;
	int isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	int jsize = (BS_3 + 2 * N3G);
	int global_id = ii * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jj * (BS_3 + 2 * N3G) + zz;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ind0 = CENT * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id_2D = ii * (BS_2 + 2 * N2G) + jj;
	int ind0 = CENT * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id_2D;
	#endif

	int zsize = 1, zoffset = 0, zlevel = 0;

	#if(N_LEVELS_1D_INT>0 && D3>0 && GPU_ENABLED==1)
	if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && j < N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n]) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if ((block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) && j >= N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = round(pow(2.0, (double)zlevel));
	zoffset = (z - N3_GPU_offset[n]) % zsize;
	#endif

	/* Constrained transport defn */

	/* Flux-ct defn */
	divE = (
		#if(N1G>1)
		(p[E1 * ksize + global_id + isize] * gdet[ind0 + (!NSY) * (BS_1 + 2 * N1G) + NSY * isize] - p[E1 * ksize + global_id - isize] * gdet[ind0 - (!NSY) * (BS_1 + 2 * N1G) - NSY * isize]) / (2.0 * _dx1)
		#endif
		#if(N2G>1)
		+ (p[E2 * ksize + global_id + jsize] * gdet[ind0 + (!NSY) + jsize * NSY] - p[E2 * ksize + global_id - jsize] * gdet[ind0 - (!NSY) - jsize * NSY]) / (2.0 * _dx2)
		#endif
		#if(N3G>1)
		+ (p[E3 * ksize + global_id + zsize] * gdet[ind0 + NSY * zsize] - p[E3 * ksize + global_id - zsize] * gdet[ind0 - NSY*zsize]) / (2.0 * (double)(zsize)*_dx3)
		#endif
	);
	return (divE / gdet[ind0]);
	#else
	return(0.0);
	#endif
}
#endif