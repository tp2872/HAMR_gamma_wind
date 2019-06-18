
#include "decs_MPI.h"

extern void raise_g(double vcov[], double gcon[][NDIM], double vcon[]);
extern void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
extern void ncov_calc(double gcon[][NDIM], double ncov[]); int Rtoprim_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);

void subcycle_rad_solve(double pb[NPR], double U[NPR], struct of_geom geom,  double Dt) {
	double factor, remainder, dU[NPR], fraction;
	int flag, keep_iterating=1, nstep=0;

	fraction = 0.25;
	remainder = 1.;

	while (keep_iterating && nstep<100) {
		source_rad(pb, &geom, dU);

		if (fraction*MY_MIN(U[UU], U[UU_RAD]) >= fabs(dU[UU_RAD] * Dt)) keep_iterating = 0;
		factor = MY_MIN(fraction*MY_MIN(U[UU], U[UU_RAD]) / fabs(dU[UU_RAD] * Dt), remainder);
		remainder -= factor;

		source_rad(pb, &geom, dU);
		U[UU_RAD] += factor*Dt*dU[UU_RAD];
		U[U1_RAD] += factor*Dt*dU[U1_RAD];
		U[U2_RAD] += factor*Dt*dU[U2_RAD];
		U[U3_RAD] += factor*Dt*dU[U3_RAD];
		U[UU] += factor*Dt*dU[UU];
		U[U1] += factor*Dt*dU[U1];
		U[U2] += factor*Dt*dU[U2];
		U[U3] += factor*Dt*dU[U3];

		flag = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pb);	
		#if(DO_FONT_FIX) 
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pb, pb[KTOT]);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pb, pb[KTOT]);
				}
			}
		}
		#endif
		#if(RAD_M1)
		if(!flag)Rtoprim(U, geom.gcov, geom.gcon, geom.g, pb, BASIC);
		#endif

		if (flag) {
			remainder += factor;
			U[UU_RAD] -= factor*Dt*dU[UU_RAD];
			U[U1_RAD] -= factor*Dt*dU[U1_RAD];
			U[U2_RAD] -= factor*Dt*dU[U2_RAD];
			U[U3_RAD] -= factor*Dt*dU[U3_RAD];
			U[UU] -= factor*Dt*dU[UU];
			U[U1] -= factor*Dt*dU[U1];
			U[U2] -= factor*Dt*dU[U2];
			U[U3] -= factor*Dt*dU[U3];
			fraction = 0.05;
		}
		nstep++;
	}
}

int implicit_rad_solve_PMHD(double pb[NPR], double U[NPR],  struct of_geom geom, double dU[NPR], double Dt){
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb[NPR],dEdpb[4][4], dEdpb_inv[4][4], bsq, errx;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating;

	// Initialize various parameters and variables:
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = 0.;
		dpb[k] = 0.;
	}
	n_iter = 0;

	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;
	while (keep_iterating) {
		//Calculate jacobian dEdpb
		for (i = UU; i < B1; i++) {
			get_state(pb_old, &geom, &q);
			mhd_calc(pb_old, 0, &q, &U_old[UU]);
			bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
			if (i == UU) {
				for (k = UU; k < B1; k++)dpb[k] = 0.;
				dpb[i] = pow(10., -9.)*(pb_old[RHO] + GAMMA*pb_old[UU] + bsq);
			}
			else {
				for (k = UU; k < B1; k++)dpb[k] = 0.;
				dpb[i] = pow(10., -5.)/sqrt(geom.gcov[i][i]);
			}
			for (k = UU; k < B1; k++) pb_new[k] = pb_old[k] + dpb[k];

			get_state(pb_new, &geom, &q);
			mhd_calc(pb_new, 0, &q, &U_new[UU]);
			get_state(pb_new, &geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);

			U_new[UU_RAD] -= (U_new[UU] - U_old[UU]);
			U_new[U1_RAD] -= (U_new[U1] - U_old[U1]);
			U_new[U2_RAD] -= (U_new[U2] - U_old[U2]);
			U_new[U3_RAD] -= (U_new[U3] - U_old[U3]);

			Rtoprim(U_new, geom.gcov, geom.gcon, geom.g, pb_new, TYPE2);

			source_rad(pb_old, &geom, dU_old);
			source_rad(pb_new, &geom, dU_new);

			for (k = UU; k < B1; k++) {
				E_old[k] = fabs((U_old[k] - U[k] - Dt*dU_old[k]));
				E_new[k] = fabs((U_new[k] - U[k] - Dt*dU_new[k]));
				dEdpb[k-1][i-UU] = (E_new[k] - E_old[k]) / dpb[i];
			}
		}

		invert_matrix(dEdpb, dEdpb_inv);

		//Tg = (GAMMA - 1.)*pb_new[UU] / pb_new[RHO];
		//error += fabs((U_new[KTOT] - U[KTOT])*Tg + Dt*dU_new[KTOT]);
		//norm += U[KTOT] * Tg;
		//error_norm = error / norm;
		for (k = 0; k < NPR; k++) pb_new[k] = pb_old[k];

		/* Make the newton step: */
		for (k = 0; k < 4; k++) {
			dpb[k+1] = -(E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
			pb_new[k+1] += dpb[k+1];
		}

		get_state(pb_new, &geom, &q);
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		get_state(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);

		U_new[UU_RAD] -= (U_new[UU] - U_old[UU]);
		U_new[U1_RAD] -= (U_new[U1] - U_old[U1]);
		U_new[U2_RAD] -= (U_new[U2] - U_old[U2]);
		U_new[U3_RAD] -= (U_new[U3] - U_old[U3]);

		Utoprim_R(U_new, geom.gcov, geom.gcon, geom.g, pb_new);

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		errx = 0.25*(fabs(dpb[UU]) / (pb_old[UU] + pb_old[RHO] + bsq) + dpb[U1] * dpb[U1] * geom.gcov[1][1] + dpb[U2] * dpb[U2] * geom.gcov[2][2] + dpb[U3] * dpb[U3] * geom.gcov[3][3]);

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(errx) <= NEWT_TOL))|| (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		else {
			for (k = 0; k < NPR; k++) pb_old[k] = pb_new[k];
		}

		n_iter++;
	}   // END of while(keep_iterating)

	if (fabs(errx) > MIN_NEWT_TOL) {
		return(1);
	}
	if (fabs(errx) <= NEWT_TOL) {
		for (k = 0; k < NPR; k++) pb[k] = pb_new[k];
		return(0);
	}

	return(0);
}

//Inversion from radiation conserved to primitive quantities
int Rtoprim(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], int lim)
{
	double U_tmp[NPR_R], prim_tmp[NPR_R];
	int i, ret;
	double alpha;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	/* Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	for (i = 0; i <= U3_RAD - E_RAD; i++) {
		U_tmp[i] = alpha * U[i + NPR_U] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	for (i = 0; i <= U3_RAD - E_RAD; i++) {
		prim_tmp[i] = prim[i + NPR_U];
	}

	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		for (i = 0; i <= U3_RAD - E_RAD; i++) {
			prim[i + NPR_U] = prim_tmp[i];
		}
	}

	return(ret);
}

int Rtoprim_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim)
{
	double Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM], Qtsq, Qdotn;
	double gammasq, y, pressure, f, ymax;
	int i;

	for (i = 0; i < 4; i++) Qcov[i] = U[i + 1];
	raise_g(Qcov, gcon, Qcon);

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	Qdotn = Qcon[0] * ncov[0]; //-Erad in McKinney2013

	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	Qsq = 0.;

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn*Qdotn; //Utilde^2 in McKinney2013

	y = Qtsq / (Qdotn*Qdotn);
	if (lim == TYPE2) {
		ymax = 1. - 0.5*(GAMMAMAX*GAMMAMAX);
		if (y > ymax) {
			Qdotn *= sqrt(ymax / y);
			y = ymax;
		}
	}
	gammasq = (2. - y + sqrt(4. - 3.*y)) / (4. - 4.*y);
	pressure = -Qdotn / (4.*gammasq - 1.);
	if (-Qdotn <= 0.) {
		prim[0] = pow(10, -300.);
		for (i = 1; i < 4; i++) prim[i] = 0;
		return 0;
	}
	else prim[0] = pressure*3.;

	for (i = 1; i < 4; i++)prim[i] = sqrt(gammasq)*Qtcon[i] / (4.*pressure*gammasq);

	if (lim == BASIC) {
		if (y <= 0.) {
			for (i = 1; i < 4; i++)prim[i] = 0.;
		}
		if (gammasq > GAMMAMAX*GAMMAMAX) {
			f = sqrt((GAMMAMAX*GAMMAMAX - 1.) / (gammasq - 1.));
			for (i = 1; i < 4; i++) prim[i] *= f;
		}
	}

	/* done! */
	return(0);
}