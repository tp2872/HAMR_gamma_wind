#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

/* this version starts w/ BL 4-velocity and
* converts to relative 4-velocity in modified
* Kerr-Schild coordinates */
void coord_transform(double *pr, int n, int ii, int jj, int zz)
{
	double X[NDIM], r, th, phi, ucon[NDIM], trans[NDIM][NDIM], tmp[NDIM], dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM], uconp[NDIM], utconp[NDIM], old[NDIM];
	double AA, BB, CC, discr;
	double alpha, gamma, beta[NDIM];
	struct of_geom geom;
	struct of_state q;
	int i, j, k, m;

	coord(n, ii, jj, zz, CENT, X);
	bl_coord(X, &r, &th, &phi);
	blgset(n, ii, jj, zz, &geom);

	ucon[1] = pr[U1];
	ucon[2] = pr[U2];
	ucon[3] = pr[U3];

	AA = geom.gcov[0][0];
	BB = 2.*(geom.gcov[0][1] * ucon[1] +
		geom.gcov[0][2] * ucon[2] +
		geom.gcov[0][3] * ucon[3]);
	CC = 1. +
		geom.gcov[1][1] * ucon[1] * ucon[1] +
		geom.gcov[2][2] * ucon[2] * ucon[2] +
		geom.gcov[3][3] * ucon[3] * ucon[3] +
		2.*(geom.gcov[1][2] * ucon[1] * ucon[2] +
		geom.gcov[1][3] * ucon[1] * ucon[3] +
		geom.gcov[2][3] * ucon[2] * ucon[3]);

	discr = BB*BB - 4.*AA*CC;
	ucon[0] = (-BB - sqrt(discr)) / (2.*AA);
	/* now we've got ucon in BL coords */
	old[1] = ucon[1];
	old[2] = ucon[2];
	old[3] = ucon[3];
	/* transform to Kerr-Schild */
	/* make transform matrix */
	DLOOP trans[j][k] = 0.;
	DLOOPA trans[j][j] = 1.;
	trans[0][1] = 2.*r / (r*r - 2.*r + a*a);
	trans[3][1] = a / (r*r - 2.*r + a*a);

	/* transform ucon */
	DLOOPA tmp[j] = 0.;
	DLOOP tmp[j] += trans[j][k] * ucon[k];
	DLOOPA ucon[j] = tmp[j];
	/* now we've got ucon in KS coords */

	/* transform to KS' coords */
	//ucon[1] *= (1. / (r - R0));
	//ucon[2] *= 1. / dxdxp[2][2];
	//ucon[3] *= 1.; //!!!ATCH: no need to transform since will use phi = X[3]
	dxdxp_func(X, dxdxp);
	/* dx^\mu/dr^\nu jacobian */
	invert_matrix(dxdxp, dxpdx);

	for (i = 0; i<NDIM; i++) {
		uconp[i] = 0;
		for (j = 0; j<NDIM; j++){
			uconp[i] += dxpdx[i][j] * ucon[j];
		}
	}
	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	get_geometry(n,ii, jj,zz, CENT, &geom);

	ucon_to_utcon(uconp, &geom, utconp);

	pr[U1] = utconp[1];
	pr[U2] = utconp[2];
	pr[U3] = utconp[3];
	//fprintf(stderr, "(%d, %d, %d) Ratio 1: %f Ratio 2: %f Ratio 3: %f \n", ii, jj, zz, utconp[1], utconp[2] / old[2], utconp[3]/old[3]);
	/* done! */
}

// These are functions needed for POSTMERGER problem
/* this version starts w/ BL 3-velocity and
 * converts to relative 4-velocity in modified
 * Kerr-Schild coordinates */
void vconbl_to_utcon(double *pr, int n, int ii, int jj, int zz)
{
  double X[NDIM], r, th, phi, vcon[NDIM], ucon[NDIM], trans[NDIM][NDIM], tmp[NDIM], dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM], uconp[NDIM], utconp[NDIM], old[NDIM];
  double AA, BB, CC, discr;
  double alpha, gamma, beta[NDIM], ut;
  struct of_geom geom;
  struct of_state q;
  int i, j, k, m;
#define USEKS (1)

  coord(n, ii, jj, zz, CENT, X);
  bl_coord(X, &r, &th, &phi);
#if(USEKS)
  ksgset(n, ii, jj, zz, &geom);
#else
  blgset(n, ii, jj, zz, &geom);
#endif

  vcon[0] = 1.0;
  vcon[1] = pr[U1];
  vcon[2] = pr[U2];
  vcon[3] = pr[U3];

  //compute u^t corresponding to the new v^i
  ut_calc_3vel(vcon, &geom, &ut);

  for(k = 0; k < NDIM; k++) {
    ucon[k] = ut * vcon[k];
  }
  /* now we've got ucon in BL coords */
  //old[1] = ucon[1];
  //old[2] = ucon[2];
  //old[3] = ucon[3];
#if(USEKS)
  //already in KS coordinates; no transformation needed
#else
  /* transform to Kerr-Schild */
  /* make transform matrix */
  DLOOP trans[j][k] = 0.;
  DLOOPA trans[j][j] = 1.;
  trans[0][1] = 2.*r / (r*r - 2.*r + a*a);
  trans[3][1] = a / (r*r - 2.*r + a*a);
  /* transform ucon */
  DLOOPA tmp[j] = 0.;
  DLOOP tmp[j] += trans[j][k] * ucon[k];
  DLOOPA ucon[j] = tmp[j];
  /* now we've got ucon in KS coords */
#endif

  /* transform to KS' coords */
  dxdxp_func(X, dxdxp);
  /* dx^\mu/dr^\nu jacobian */
  invert_matrix(dxdxp, dxpdx);

  for (i = 0; i<NDIM; i++) {
    uconp[i] = 0;
    for (j = 0; j<NDIM; j++){
      uconp[i] += dxpdx[i][j] * ucon[j];
    }
  }
  /* now solve for v-- we can use the same u^t because
   * it didn't change under KS -> KS' */
  get_geometry(n,ii, jj,zz, CENT, &geom);

  ucon_to_utcon(uconp, &geom, utconp);

  pr[U1] = utconp[1];
  pr[U2] = utconp[2];
  pr[U3] = utconp[3];
  //fprintf(stderr, "(%d, %d, %d) Ratio 1: %f Ratio 2: %f Ratio 3: %f \n", ii, jj, zz, utconp[1], utconp[2] / old[2], utconp[3]/old[3]);
  /* done! */
}

/* This function takes Utilde 3-velocity and
 * transforms it into 4-velocity in modified Kerr-Schild coordinates
 */
void utilde_to_ucon(double *pr, double udphi, double mudt, int n, int ii, int jj, int zz, double tilt)
{
  double X[NDIM], r, th, phi, pos_new[NDIM], X_cart[NDIM], vtcon[NDIM], utcon[NDIM], trans[NDIM][NDIM], tmp[NDIM], dxdr[NDIM][NDIM], drdx[NDIM][NDIM], dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM], uconp[NDIM], utconp[NDIM], utconp_new[NDIM], ucon[NDIM], ucov[NDIM];
  double AA, BB, CC, discr, udphi_new, err, err_tol;
  double alpha, gamma, beta[NDIM], ut;
  struct of_geom geom;
  struct of_state q;
  int i, j, k, m, max_iter;
#define USEKS (1)

  coord(n, ii, jj, zz, CENT, X);
  bl_coord(X, &r, &th, &phi);

  pos_new[1] = r;
  pos_new[2] = th;
  pos_new[3] = phi;
  #if (TILTED)
  sph_to_cart(X_cart, &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
  rotate_coord(X_cart, -tilt);
  cart_to_sph(X_cart, &r, &th, &phi);
  #endif

#if(USEKS)
  ksgset(n, ii, jj, zz, &geom);
#else
  blgset(n, ii, jj, zz, &geom);
#endif

  // By definition, U^t tilde = 0
  vtcon[0] = 0.0;
  vtcon[1] = pr[U1];
  vtcon[2] = pr[U2];
  vtcon[3] = pr[U3];

  #if (TILTED)
  rotate_vector(vtcon, pos_new, &r, &th, &phi, tilt);
  #endif

  /* transform to KS' coords */
  dxdxp_func(X, dxdxp);
  /* dx^\mu/dr^\nu jacobian */
  invert_matrix(dxdxp, dxpdx);

  #if (!BHNSQ2_1)
  /* Jacobian transformation from spherical to cartesian coords */
  dxdr_sph_to_cart(r, th, phi, dxdr);
  invert_matrix(dxdr, drdx);
  // converts the input Utilde^{x,y,z} to Utilde^{r,th,phi}
  utcon[0] = 0.0;
  for (i = 1; i<NDIM; i++) {
    utcon[i] = 0;
    for (j = 0; j<NDIM; j++){
      utcon[i] += drdx[i][j] * vtcon[j];
    }
  }
  #else
  utcon[0] = 0.0;
  utcon[1] = vtcon[1];
  utcon[2] = vtcon[2];
  utcon[3] = vtcon[3];
  #endif

#if(USEKS)
  //already in KS coordinates; no transformation needed
#else
  /* transform to Kerr-Schild */
  /* make transform matrix */
  DLOOP trans[j][k] = 0.;
  DLOOPA trans[j][j] = 1.;
  trans[0][1] = 2.*r / (r*r - 2.*r + a*a);
  trans[1][0] = trans[0][1];
  trans[3][1] = a / (r*r - 2.*r + a*a);
  trans[1][3] = trans[3][1];
  DLOOPA tmp[j] = utcon[j];
  utcon[0] = 0.0;
	for (i = 1; i < NDIM; i++) {
		utcon[i] = 0;
		for (j = 0; j < NDIM; j++) {
			utcon[i] += trans[i][j] * tmp[j];
		}
	}

#endif

	#if (0)
	// 1. Find u^t from SpEC velocities
	double vconp[NDIM], ucon[NDIM], ucov[NDIM];
	pr[U1] = utcon[1];
	pr[U2] = utcon[2];
	pr[U3] = utcon[3];
	ucon_calc(pr, &geom, ucon);

	// 2. Find ucon1 and ucon3 given mudt and udphi from SpEC
	double b1 = udphi - geom.gcov[3][0] * ucon[0];
	double b2 = -mudt - geom.gcov[0][0] * ucon[0];
	double detA = geom.gcov[3][1] * geom.gcov[0][3] - geom.gcov[0][1] * geom.gcov[3][3];
	ucon[1] = (geom.gcov[0][3] * b1 - geom.gcov[3][3] * b2) / detA;
	ucon[3] = (-geom.gcov[0][1] * b1 + geom.gcov[3][1] * b2) / detA;

	// 3. Compute ucon2 from ucon0, ucon1 and ucon3 from u^2 = -1
	ucon[2] = (- 1.0 - geom.gcov[0][0] * ucon[0] * ucon[0] - geom.gcov[1][1] * ucon[1] * ucon[1] - geom.gcov[3][3] * ucon[3] * ucon[3] - 2.0 * geom.gcov[0][1] * ucon[0] * ucon[1] - 2.0 * geom.gcov[0][3] * ucon[0] * ucon[3] - 2.0 * geom.gcov[1][3] * ucon[1] * ucon[3]) /(geom.gcov[2][2]);
	ucon[2] = sqrt(ucon[2]);
	//ucon[2] = 0.0;
	// 4. Convert ucon to u^tilde
	ucon_to_utcon(ucon, &geom, utcon);
	#endif

  // converts Utilde^{r,th,phi} from the previous loop into Utilde^{x1,x2,x3}
  utconp[0] = 0.0;
  for (i = 1; i<NDIM; i++) {
    utconp[i] = 0;
    for (j = 0; j<NDIM; j++){
      utconp[i] += dxpdx[i][j] * utcon[j];
    }
  }

  pr[U1] = utconp[1];
  pr[U2] = utconp[2];
  pr[U3] = utconp[3];

  /* now solve for v-- we can use the same u^t because
   * it didn't change under KS -> KS' */

  // This calculates Utilde^{phi} given u_{phi}. Iterations are required because the relation between them is not linear.


// Commented out this part of the code that modifies utconp by matching udphi's
#if (0)
  max_iter = 50;
  err_tol = 1.0E-4;
  for (i = 0; i < max_iter; i++) {
    udphi_to_utuphi(utconp, udphi, &udphi_new, &geom, utconp_new);
    DLOOPA utconp[j] = utconp_new[j];
    err = fabs(2 * (udphi_new - udphi)/(udphi + udphi_new + 1.0E-7));
    if (err <= err_tol) break;
    udphi = udphi_new;
  }
  
  pr[U1] = utconp[1];
  pr[U2] = utconp[2];
  pr[U3] = utconp[3];
#endif

  /* done! */
}

// This function calculates Utilde^{phi} from Utilde_{phi}
void udphi_to_utuphi(double *ucon, double udphi, double *udphi_new, struct of_geom *geom, double *utcon)
{
  double alpha, beta[NDIM], gamma, gamma_new, AA, BB; //, A, B, C, D, E, F u_minus, u_plus;
  int j, k;

  /* now solve for v-- we can use the same u^t because
   * it didn't change under KS -> KS' */
  alpha = 1. / sqrt(-geom->gcon[0][0]);
  SLOOPA beta[j] = geom->gcon[0][j] * alpha*alpha;
  gamma = alpha*ucon[0];

  utcon[0] = 0;
  SLOOPA utcon[j] = ucon[j];

  SLOOPA utcon[3] = - geom->gcov[3][j] * beta[j] * gamma / alpha;
  utcon[3] += geom->gcov[3][0] * gamma / alpha + udphi - (geom->gcov[3][1] * ucon[1] + geom->gcov[3][2] * ucon[2]);
  utcon[3] /= (geom->gcov[3][3]);

  // Update udphi

  AA = - geom->gcov[0][0] * geom->gcon[0][0];
  BB = 0.0;
  SLOOP AA += geom->gcov[j][k] * beta[j] * beta[k] / (alpha * alpha);
  SLOOP BB += 2 * geom->gcov[j][k] * utcon[j] * beta[k] / alpha;
  gamma_new = BB / (1 + AA);

  *udphi_new = geom->gcov[3][0] * gamma_new / alpha;
  SLOOPA *udphi_new += geom->gcov[3][j] * (utcon[j] - gamma_new * beta[j] / alpha);
}

void dxdr_sph_to_cart(double r, double th, double phi, double dxdr[][NDIM])
  {
  	int j;
	//double Xh[NDIM], Xl[NDIM];
	//double Vh[NDIM], Vl[NDIM];
	for(j = 1; j < NDIM; j++) {
		dxdr[0][j] = 0.0;
		dxdr[j][0] = 0.0;
	}
	dxdr[0][0] = 1.0;

	dxdr[1][1] = sin(th) * cos(phi);
	dxdr[1][2] = r * cos(th) * cos(phi);
	dxdr[1][3] = - r * sin(th) * sin(phi);

	dxdr[2][1] = sin(th) * sin(phi);
	dxdr[2][2] = r * cos(th) * sin(phi);
	dxdr[2][3] = r * sin(th) * cos(phi);

	dxdr[3][1] = cos(th);
	dxdr[3][2] = - r * sin(th);
	dxdr[3][3] = 0.0;
  }

//Transform coordinates to Cartesian
void sph_to_cart(double X[NDIM], double *r, double *th, double *phi){
	X[1] = r[0] * sin(th[0])*cos(phi[0]);
	X[2] = r[0] * sin(th[0])*sin(phi[0]);
	X[3] = r[0] * cos(th[0]);
}

//Rotate by angle tilt around y-axis, see wikipedia
void rotate_coord(double X[NDIM], double tilt){
	double X_tmp[NDIM];
	int i;
	for (i = 1; i < NDIM; i++){
		X_tmp[i] = X[i];
	}
	X[1] = X_tmp[1] * cos(tilt) + X_tmp[3] * sin(tilt);
	X[2] = X_tmp[2];
	X[3] = -X_tmp[1] * sin(tilt) + X_tmp[3] * cos(tilt);
}

//Transform coordinates back to spherical
void cart_to_sph(double X[NDIM], double *r, double *th, double *phi){
	r[0] = sqrt(X[1] * X[1] + X[2] * X[2] + X[3] * X[3]);
	th[0] = acos(X[3] / r[0]);
	phi[0] = atan2(X[2], X[1]);
}

/*Calculates covariant vector components after vector is rotated from (r, th, phi) to (pos_new[1], pos_new[2], pos_new[3]) over angle tilt*/
void rotate_vector(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt){
	#if(0)
		double bl_gcov[NDIM][NDIM], gdet1, gdet2;
		double V_tmp[NDIM], X_tmp[NDIM], pos_new_tmp[NDIM];
		int i;
		for (i = 1; i < NDIM; i++){
			V_tmp[i] = V[i];
			pos_new_tmp[i] = pos_new[i];
		}

		#if (WHICHPROBLEM == POSTMERGER_PROBLEM)
		kerr_gcov_func(*r, *th, bl_gcov);
		#else
		bl_gcov_func(*r, *th, bl_gcov);
		#endif

		V_tmp[1] *= sqrt(bl_gcov[1][1]);
		V_tmp[2] *= sqrt(bl_gcov[2][2]);
		V_tmp[3] *= sqrt(bl_gcov[3][3]);
		//V_tmp[3] = sqrt(bl_gcov[3][3] * V_tmp[3] * V_tmp[3]+2.*bl_gcov[0][3] * V_tmp[0] * V_tmp[3]);

		X_tmp[1] = V_tmp[1] * sin(*th)*cos(*phi) + V_tmp[2] * cos(*th)*cos(*phi) - V_tmp[3] * sin(*phi);
		X_tmp[2] = V_tmp[1] * sin(*th)*sin(*phi) + V_tmp[2] * cos(*th)*sin(*phi) + V_tmp[3] * cos(*phi);
		X_tmp[3] = V_tmp[1] * cos(*th) - V_tmp[2] * sin(*th);

		rotate_coord(X_tmp, tilt);

		#if (WHICHPROBLEM == POSTMERGER_PROBLEM)
		kerr_gcov_func(pos_new[1], pos_new[2], bl_gcov);
		#else
		bl_gcov_func(pos_new[1], pos_new[2], bl_gcov);
		#endif

		//gdet2 = gdet_func(bl_gcov);
		V[0] = V_tmp[0];
		V[1] = (X_tmp[1] * sin(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * sin(pos_new[2])*sin(pos_new[3]) + X_tmp[3] * cos(pos_new[2])) / sqrt(bl_gcov[1][1]);
		V[2] = (X_tmp[1] * cos(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * cos(pos_new[2])*sin(pos_new[3]) - X_tmp[3] * sin(pos_new[2])) / sqrt(bl_gcov[2][2]);
		V[3] = (-X_tmp[1] * sin(pos_new[3]) + X_tmp[2] * cos(pos_new[3])) / sqrt(bl_gcov[3][3]);
		//V[3] = (-bl_gcov[0][3] * V[0] + V[3] / fabs(V[3])*sqrt(pow(bl_gcov[0][3] * V[0], 2.) + bl_gcov[3][3]*pow(V[3],2.))) / bl_gcov[3][3];
	#else
	double dxdxt[NDIM][NDIM], dxtdx[NDIM][NDIM], Vp[NDIM], dxdr[NDIM][NDIM], drdx[NDIM][NDIM], V_tmp[NDIM], V_new[NDIM];
	int i, j;

	for (i = 1; i < NDIM; i++) {
		V_tmp[i] = V[i];
	}

	//compute Jacobian nt->t (dt/dnt)
	dxdxt[0][0] = 1.;
	dxdxt[0][1] = 0.;
	dxdxt[0][2] = 0.;
	dxdxt[0][3] = 0.;
	dxdxt[1][0] = 0.;
	dxdxt[1][1] = cos(tilt);
	dxdxt[1][2] = 0.;
	dxdxt[1][3] = sin(tilt);
	dxdxt[2][0] = 0.;
	dxdxt[2][1] = 0.;
	dxdxt[2][2] = 1.;
	dxdxt[2][3] = 0.;
	dxdxt[3][0] = 0.;
	dxdxt[3][1] = -sin(tilt);
	dxdxt[3][2] = 0.0;
	dxdxt[3][3] = cos(tilt);
	invert_matrix(dxdxt, dxtdx);

	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(th[0]) * cos(phi[0]);
	dxdr[1][2] = r[0] * cos(th[0]) * cos(phi[0]);
	dxdr[1][3] = -r[0] * sin(th[0]) * sin(phi[0]);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(th[0]) * sin(phi[0]);
	dxdr[2][2] = r[0] * cos(th[0]) * sin(phi[0]);
	dxdr[2][3] = r[0] * sin(th[0]) * cos(phi[0]);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(th[0]);
	dxdr[3][2] = -r[0] * sin(th[0]);
	dxdr[3][3] = 0.;

	//convert from kerr schild to cartesian coordinates
	for (i = 0; i < NDIM; i++) {
		V[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V[i] += dxdr[i][j] * V_tmp[j];
		}
	}

	//convert from cartesian to tilted cartesian coordinates
	for (i = 0; i < NDIM; i++) {
		V_tmp[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V_tmp[i] += dxdxt[i][j] * V[j];
		}
	}
	
	//compute Jacobian x1,x2,x3 -> r,th,phi (dr/dx1)
	Vp[1] = r[0] * sin(th[0]) * cos(phi[0]);
	Vp[2] = r[0] * sin(th[0]) * sin(phi[0]);
	Vp[3] = r[0] * cos(th[0]);

	V_new[1] = Vp[1] * cos(tilt) + Vp[3] * sin(tilt);
	V_new[2] = Vp[2];
	V_new[3] = -sin(tilt) * Vp[1] + cos(tilt) * Vp[3];
	Vp[1] = sqrt(V_new[1] * V_new[1] + V_new[2] * V_new[2] + V_new[3] * V_new[3]);
	Vp[2] = acos(V_new[3] / Vp[1]);
	Vp[3] = atan2(V_new[2], V_new[1]);
	//fprintf(stderr, "tilt: %f, r: %f/%f theta: %f/%f, phi: %f/%f \n", tilt, pos_new[1], Vp[1], pos_new[2], Vp[2], pos_new[3], Vp[3]);
	if (Vp[2] < 0.0) Vp[2] *= -1;
	if (Vp[2] > M_PI) Vp[2] = M_PI - (Vp[2] - M_PI);

	#if(COORDSINGFIX)
	if (fabs(Vp[2]) < SINGSMALL) {
		if (Vp[2] >= 0.0) Vp[2] = SINGSMALL;
		if (Vp[2] < 0.0)  Vp[2] = -SINGSMALL;
	}
	if (fabs(M_PI - Vp[2]) < SINGSMALL) {
		if (Vp[2] >= M_PI) Vp[2] = M_PI + SINGSMALL;
		if (Vp[2] < M_PI)  Vp[2] = M_PI - SINGSMALL;
	}
	#endif

	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(Vp[2])*cos(Vp[3]);
	dxdr[1][2] = r[0]*cos(Vp[2])*cos(Vp[3]);
	dxdr[1][3] = -r[0]*sin(Vp[2])*sin(Vp[3]);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(Vp[2])*sin(Vp[3]);
	dxdr[2][2] = r[0]*cos(Vp[2])*sin(Vp[3]);
	dxdr[2][3] = r[0]*sin(Vp[2])*cos(Vp[3]);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(Vp[2]);
	dxdr[3][2] = -r[0]*sin(Vp[2]);
	dxdr[3][3] = 0.;
	invert_matrix(dxdr, drdx);

	//convert back to tilted kerr-schild coordinates
	for (i = 0; i < NDIM; i++) {
		V[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V[i] += drdx[i][j] * V_tmp[j];
		}
	}
	#endif
}

/*Calculates covariant vector components after vector is rotated from (r, th, phi) to (pos_new[1], pos_new[2], pos_new[3]) over angle tilt*/
void rotate_vector2(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt){
	#if(0)
	double bl_gcov[NDIM][NDIM], bl_gcon[NDIM][NDIM], bl_gcon1[NDIM][NDIM], bl_gcon2[NDIM][NDIM], bl_gcov1[NDIM][NDIM], bl_gcov2[NDIM][NDIM], dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM], gdet1, gdet2;
	double V_tmp[NDIM], X[NDIM], X_tmp[NDIM], pos_new_tmp[NDIM];
	double theta_solve, theta_old, derivative;
	double delta_X2 = 0.1*M_PI / (double)N2*2. / M_PI;
	int step = 0;
	int i, j, k, l;
	for (i = 1; i < NDIM; i++){
		V_tmp[i] = V[i];
		pos_new_tmp[i] = pos_new[i];
	}

	/*Calculate length of vector wrt orthonormal basis instead of coordinate basis*/
	X[1] = log(*r - RB);
	X[2] = 2. / M_PI*(*th) - 1.;
	X[3] = *phi;
	/*do{
		bl_coord(X, &(*r), &(theta_solve), &(*phi));
		theta_solve -= *th;
		theta_old = theta_solve;
		X[2] += delta_X2;
		bl_coord(X, &(*r), &(theta_solve), &(*phi));
		theta_solve -= *th;
		derivative = (theta_solve - theta_old) / delta_X2;
		X[2] -= theta_solve / derivative;
		step++;
	} while (fabs(theta_solve)>2.*M_PI / (double)N2/10. && step<30);*/
	kerr_gcov_func(*r, *th, bl_gcov);
	if (invert_matrix(bl_gcov, bl_gcon))fprintf(stderr, "Rotate error 0 %f %f\n", *r, *th);
	dxdxp_func2(X, dxdxp);
	if (invert_matrix(dxdxp, dxpdx))fprintf(stderr, "Rotate error 1 \n");

	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			bl_gcon1[i][j] = 0;
			bl_gcov1[i][j] = 0;

			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					bl_gcon1[i][j] += bl_gcon[k][l] * dxpdx[i][k] * dxpdx[j][l];
					bl_gcov1[i][j] += bl_gcov[k][l] * dxdxp[k][i] * dxdxp[l][j];

				}
			}
		}
	}
	gdet1 = gdet_func(bl_gcov1);
	V_tmp[1] *= sqrt(fabs(bl_gcon1[1][1]));
	V_tmp[2] *= sqrt(fabs(bl_gcon1[2][2]));
	V_tmp[3] *= sqrt(fabs(bl_gcon1[3][3]));

	/*Calculate Cartesian components (x, y, z) at pos_newition (r, th, phi) of vector V*/
	X_tmp[1] = V_tmp[1] * sin(*th)*cos(*phi) + V_tmp[2] * cos(*th)*cos(*phi) - V_tmp[3] * sin(*phi);
	X_tmp[2] = V_tmp[1] * sin(*th)*sin(*phi) + V_tmp[2] * cos(*th)*sin(*phi) + V_tmp[3] * cos(*phi);
	X_tmp[3] = V_tmp[1] * cos(*th) - V_tmp[2] * sin(*th);

	/*Rotate vector over angle tilt around y-axis*/
	rotate_coord(X_tmp, tilt);

	/*Tranform vector back to coordinate basis (r, th, phi) at pos_newition (pos_new[1], pos_new[2], pos_new[3])*/
	X[1] = log(pos_new[1] - RB);
	X[2] = 2. / M_PI*pos_new[2] - 1.;
	X[3] = pos_new[3];
	step = 0;
	/*do{
		bl_coord(X, &(pos_new[1]), &(theta_solve), &(pos_new[3]));
		theta_solve -= pos_new[2];
		theta_old = theta_solve;
		X[2] += delta_X2;
		bl_coord(X, &(pos_new[1]), &(theta_solve), &(pos_new[3]));
		theta_solve -= pos_new[2];
		derivative = (theta_solve - theta_old) / delta_X2;
		X[2] -= theta_solve / derivative;
		step++;
	} while (fabs(theta_solve)>2.*M_PI / (double)N2/10. && step<30);*/
	kerr_gcov_func(pos_new[1], pos_new[2], bl_gcov);
	if (invert_matrix(bl_gcov, bl_gcon))fprintf(stderr, "Rotate error 2 \n");

	dxdxp_func2(X, dxdxp);
	if (invert_matrix(dxdxp, dxpdx))fprintf(stderr, "Rotate error 3 \n");

	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			bl_gcon2[i][j] = 0;
			bl_gcov2[i][j] = 0;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					bl_gcon2[i][j] += bl_gcon[k][l] * dxpdx[i][k] * dxpdx[j][l];
					bl_gcov2[i][j] += bl_gcov[k][l] * dxdxp[k][i] * dxdxp[l][j];
				}
			}
		}
	}
	V[1] = (X_tmp[1] * sin(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * sin(pos_new[2])*sin(pos_new[3]) + X_tmp[3] * cos(pos_new[2]))/ sqrt(bl_gcon2[1][1]);
	V[2] = (X_tmp[1] * cos(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * cos(pos_new[2])*sin(pos_new[3]) - X_tmp[3] * sin(pos_new[2]))/ sqrt(bl_gcon2[2][2]);
	V[3] = (-X_tmp[1] * sin(pos_new[3]) + X_tmp[2] * cos(pos_new[3]))/ sqrt(bl_gcon2[3][3]);
	#else
	double dxdxt[NDIM][NDIM], dxtdx[NDIM][NDIM], Vp[NDIM], dxdr[NDIM][NDIM], drdx[NDIM][NDIM], V_tmp[NDIM], V_new[NDIM];
	int i, j;

	for (i = 1; i < NDIM; i++) {
		V_tmp[i] = V[i];
	}

	//compute Jacobian nt->t (dt/dnt)
	dxdxt[0][0] = 1.;
	dxdxt[0][1] = 0.;
	dxdxt[0][2] = 0.;
	dxdxt[0][3] = 0.;
	dxdxt[1][0] = 0.;
	dxdxt[1][1] = cos(tilt);
	dxdxt[1][2] = 0.;
	dxdxt[1][3] = sin(tilt);
	dxdxt[2][0] = 0.;
	dxdxt[2][1] = 0.;
	dxdxt[2][2] = 1.;
	dxdxt[2][3] = 0.;
	dxdxt[3][0] = 0.;
	dxdxt[3][1] = -sin(tilt);
	dxdxt[3][2] = 0.0;
	dxdxt[3][3] = cos(tilt);
	invert_matrix(dxdxt, dxtdx);

	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(th[0]) * cos(phi[0]);
	dxdr[1][2] = r[0] * cos(th[0]) * cos(phi[0]);
	dxdr[1][3] = -r[0] * sin(th[0]) * sin(phi[0]);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(th[0]) * sin(phi[0]);
	dxdr[2][2] = r[0] * cos(th[0]) * sin(phi[0]);
	dxdr[2][3] = r[0] * sin(th[0]) * cos(phi[0]);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(th[0]);
	dxdr[3][2] = -r[0] * sin(th[0]);
	dxdr[3][3] = 0.;
	invert_matrix(dxdr, drdx);

	//convert from kerr schild to cartesian coordinates
	for (i = 0; i < NDIM; i++) {
		V[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V[i] += drdx[j][i] * V_tmp[j];
		}
	}

	//convert from cartesian to tilted cartesian coordinates
	for (i = 0; i < NDIM; i++) {
		V_tmp[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V_tmp[i] += dxtdx[j][i] * V[j];
		}
	}
	
	//compute Jacobian x1,x2,x3 -> r,th,phi (dr/dx1)
	Vp[1] = r[0] * sin(th[0]) * cos(phi[0]);
	Vp[2] = r[0] * sin(th[0]) * sin(phi[0]);
	Vp[3] = r[0] * cos(th[0]);

	V_new[1] = Vp[1] * cos(tilt) + Vp[3] * sin(tilt);
	V_new[2] = Vp[2];
	V_new[3] = -sin(tilt) * Vp[1] + cos(tilt) * Vp[3];
	Vp[1] = sqrt(V_new[1] * V_new[1] + V_new[2] * V_new[2] + V_new[3] * V_new[3]);
	Vp[2] = acos(V_new[3] / Vp[1]);
	Vp[3] = atan2(V_new[2], V_new[1]);
	//fprintf(stderr, "tilt: %f, r: %f/%f theta: %f/%f, phi: %f/%f \n", tilt, pos_new[1], Vp[1], pos_new[2], Vp[2], pos_new[3], Vp[3]);
	if (Vp[2] < 0.0) Vp[2] *= -1;
	if (Vp[2] > M_PI) Vp[2] = M_PI - (Vp[2] - M_PI);

	#if(COORDSINGFIX)
	if (fabs(Vp[2]) < SINGSMALL) {
		if (Vp[2] >= 0.0) Vp[2] = SINGSMALL;
		if (Vp[2] < 0.0)  Vp[2] = -SINGSMALL;
	}
	if (fabs(M_PI - Vp[2]) < SINGSMALL) {
		if (Vp[2] >= M_PI) Vp[2] = M_PI + SINGSMALL;
		if (Vp[2] < M_PI)  Vp[2] = M_PI - SINGSMALL;
	}
	#endif

	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(Vp[2])*cos(Vp[3]);
	dxdr[1][2] = r[0]*cos(Vp[2])*cos(Vp[3]);
	dxdr[1][3] = -r[0]*sin(Vp[2])*sin(Vp[3]);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(Vp[2])*sin(Vp[3]);
	dxdr[2][2] = r[0]*cos(Vp[2])*sin(Vp[3]);
	dxdr[2][3] = r[0]*sin(Vp[2])*cos(Vp[3]);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(Vp[2]);
	dxdr[3][2] = -r[0]*sin(Vp[2]);
	dxdr[3][3] = 0.;

	//convert back to tilted kerr-schild coordinates
	for (i = 0; i < NDIM; i++) {
		V[i] = 0;
		for (j = 0; j < NDIM; j++) {
			V[i] += dxdr[j][i] * V_tmp[j];
		}
	}
	#endif
}

void elliptical_coord(double X_cart[NDIM], double pos_new[NDIM], double *r, double eccentricity){
	double vu = pos_new[3];
	double a_axis = *r; //semi-major axis
	*r = fabs(pos_new[1] * (1. + eccentricity*cos(vu)) / (1. - pow(eccentricity, 2.))); //circular radius r_old corresponding to elliptical radius r_new
	return;
}

void elliptical_vector(double X_cart[NDIM], double V_old[NDIM], double V_new[NDIM], double pos_new[NDIM], double *r, double *th, double eccentricity){
	double bl_gcov[NDIM][NDIM];
	double vu = pos_new[3];
	double a_axis = *r; //semi-major axis
	double b_axis = a_axis*sqrt(1. - pow(eccentricity, 2.)); //semi-minor axis
	double period = pow(pow(a_axis, 3.)*4.*pow(M_PI, 2.), 0.5);
	//convert from coordinate basis to ~orthonormal basis
	bl_gcov_func(*r, *th, bl_gcov);
	V_old[3] *= sqrt(bl_gcov[3][3]);
	double slowdown_factor = V_old[3] * sqrt(*r); //calculate how sub-keplerian the flow is
	V_new[3] = slowdown_factor*a_axis*b_axis*2.*M_PI / (period * pow(pos_new[1], 2.)); //calculate new toroidal velocity component
	double p = a_axis*(1. - pow(eccentricity, 2.));
	V_new[1] = p*eccentricity*V_new[3] * sin(vu) / pow(1. + eccentricity*cos(vu), 2.); //calculate new radial velocity component
	V_new[2] = 0.;
	V_old[3] /= sqrt(bl_gcov[3][3]);
}

void init_neutrinos(double ph[NPR]) {
#if (NEUTRINOS_M1)
	double F2, F3, mu_nu, mu_p, mu_n, mu_e;
	eos_mode_rhotemp_etaele(ph[RHO], ph[UU], ph[YE], &mu_e
		#if (DONUCLEAR)
		, ph[XALPHA], ph[XATM]
		#endif
	);
	calc_mu_np(ph[RHO], ph[UU], 1.0 - ph[YE], ph[YE], &mu_n, &mu_p);
	mu_nu = mu_p + mu_e - mu_n + (MP_CGS + ME_CGS - MN_CGS) * C_CGS * C_CGS / (BOLTZ_CGS * ph[UU]);
	double T_nu;
	for (int sp = 0; sp < NU_SPECIES; sp++) {
		if (sp == 0) {
			F2 = calc_fermiint2(mu_nu);
			F3 = calc_fermiint3(mu_nu);
		}
		else if (sp == 1) {
			F2 = calc_fermiint2(-mu_nu);
			F3 = calc_fermiint3(-mu_nu);
		}
		else if (sp == 2) {
			F2 = calc_fermiint2(0.0);
			F3 = calc_fermiint3(0.0);
		}

		//if (ph[RHO] > pow(10., nulib_dlo) || ph[YE] < nulib_yhi || ph[YE] > nulib_ylo) {
		//	// Energy density
		//	ph[index_nu(UU_NU, sp)] = 8. * M_PI * pow(BOLTZ_CGS * ph[UU], 4.) / pow(PLANCK_CGS * C_CGS, 3.) * F3 / (ENERGY_DENSITY_SCALE);
		//	ph[index_nu(UU_NU, sp)] = MY_MAX(ph[index_nu(UU_NU, sp)], 1e-30);
		//
		//	// Number density
		//	ph[index_nu(NUMBER_NU, sp)] = 8. * M_PI * pow(BOLTZ_CGS * ph[UU], 3.) / pow(PLANCK_CGS * C_CGS, 3.) * F2 / MASS_DENSITY_SCALE;
		//	ph[index_nu(NUMBER_NU, sp)] = MY_MAX(ph[index_nu(NUMBER_NU, sp)], 1e-30);
		//}
		//else {
		ph[index_nu(UU_NU, sp)] = 1e-30;
		T_nu = pow(ph[index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		ph[index_nu(NUMBER_NU, sp)] = ph[index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * T_nu); //1e-30;
		//}

		ph[index_nu(U1_NU, sp)] = ph[U1];
		ph[index_nu(U2_NU, sp)] = ph[U2];
		ph[index_nu(U3_NU, sp)] = ph[U3];
	}

#endif
}

void init_nuclear(double ph[NPR]) {
#if (DONUCLEAR)
	#if (DOHELM_TEMPERATURE != 1)
	fprintf(stderr, "Won't work without Tgas as a primitive variable! Exiting...");
	exit(1);
	#endif

	double x_n, x_p;
	// Compute abundances 
	if (ph[XATM] < x_atm_cutoff && ph[UU] > tgas_cutoff) {
		ph[XATM] = 0.0;
		nse_abundances(ph[RHO] * MASS_DENSITY_SCALE, ph[UU], ph[YE], &x_n, &x_p, &ph[XALPHA]);
	}
	else {
		x_n = get_xn(ph[YE], ph[XALPHA]);
		x_p = get_xp(ph[YE], ph[XALPHA]);
		// normalize
		double x_sum = x_n + x_p + ph[XALPHA] + ph[XATM];
		if (x_sum > 1.0) {
			ph[XALPHA] = ph[XALPHA] / x_sum;
			ph[XATM] = ph[XATM] / x_sum;
		}
	}

	fprintf(stderr, "\t\n xn, xp, xa, xatm, ye = %e %e %e %e %e", x_n, x_p, ph[XALPHA], ph[XATM], ph[YE]);

	// Check if the abundances are out of bounds
	ph[XALPHA] = MY_MIN(1.0, ph[XALPHA]);
	ph[XATM] = MY_MIN(1.0, ph[XATM]);

	ph[XALPHA] = MY_MAX(1e-10, ph[XALPHA]);
	ph[XATM] = MY_MAX(1e-10, ph[XATM]);

	return;
#endif
}

#if(TWO_T)
void set_2T_entropy(double pi[NPR], double bsq) {
	double delta, delta_f=0.5, p_tot, pe_new, pi_new, pe_old, gamg, Theta_e, Theta_i, game, gami, ue, ui, error_0, error_1, derror_dpe, errx, offset = 1.e-8;
	int keep_iterating = 1, i, n_iter = 0;

	//Set desired (total) gas pressure
	p_tot = (GAMMA - 1.0) * pi[UU];

	//Set initial guess for electron pressure
	pe_new = p_tot * delta_f;
	pe_old = pe_new;

	//Iterate electron pressure, minimize error in real vs predicted deltaf
	while (keep_iterating) {
		//Calculate ion pressure from total pressure
		pi_new = p_tot - pe_new;

		//Calculate game and gami
		#if(CONSTANTGAMMA || FIXEDGAMMA)
		game = GAMMAE;
		gami = GAMMA;
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		game = (10.0 + 20.0 * pe_new / pi[RHO] * MU_E * MASS_RATIO) / (6.0 + 15.0 * pe_new / pi[RHO] * MU_E * MASS_RATIO);
		gami = (10.0 + 20.0 * pi_new / pi[RHO] * MU_I) / (6.0 + 15.0 * pi_new / pi[RHO] * MU_I);
		#endif

		//Calculate ue and ui
		ue = pe_new / (game - 1.0);
		ui = pi_new / (gami - 1.0);

		//Calculate electron and ion entropy primitive variables delta_f
		#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15, Ryan+17
			#if(FULL_ENTROPY)
			pi[ENTRE] = 1.0 / (GAMMAE - 1.) * log(pe_new * pow(pi[RHO], -GAMMAE));
			pi[ENTRI] = 1.0 / (GAMMA - 1.) * log(pi_new * pow(pi[RHO], -GAMMA));
			#else
			pi[ENTRE] = pe_new * pow(pi[RHO], -GAMMAE);
			pi[ENTRI] = pi_new * pow(pi[RHO], -GAMMA);
			#endif
		#elif(VARGAMMA)   // variable gamma: Sadowski+17, Chael+19
		Theta_e = pe_new / pi[RHO] * MU_E * MASS_RATIO;
		Theta_i = pi_new / pi[RHO] * MU_I;
			#if(FULL_ENTROPY_VARGAMMA)
			pi[ENTRE] = log(pow(Theta_e, 1.5) * pow(Theta_e + 0.4, 1.5) / pi[RHO]);
			pi[ENTRI] = log(pow(Theta_i, 1.5) * pow(Theta_i + 0.4, 1.5) / pi[RHO]);
			#else
			pi[ENTRE] = Theta_e * (Theta_e + 0.4) / pow(pi[RHO], 2. / 3.);
			pi[ENTRI] = Theta_i * (Theta_i + 0.4) / pow(pi[RHO], 2. / 3.);
			#endif	
		#endif	
	
		//Calculate total error in delta
		delta = ue / (ue + ui);
		delta_f = calc_delta(pi, bsq);   // initial Tel/Ttot (temperature ratio)
		error_0 = delta_f - delta;

		//Calculate ion pressure from total pressure
		pe_new = (1.0 + offset) * pe_new;
		pi_new = p_tot - pe_new;

		//Calculate game and gami
		#if(CONSTANTGAMMA || FIXEDGAMMA)
		game = GAMMAE;
		gami = GAMMA;
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		game = (10.0 + 20.0 * pe_new / pi[RHO] * MU_E * MASS_RATIO) / (6.0 + 15.0 * pe_new / pi[RHO] * MU_E * MASS_RATIO);
		gami = (10.0 + 20.0 * pi_new / pi[RHO] * MU_I) / (6.0 + 15.0 * pi_new / pi[RHO] * MU_I);
		#endif

		//Calculate ue and ui
		ue = pe_new / (game - 1.0);
		ui = pi_new / (gami - 1.0);

		//Calculate electron and ion entropy primitive variables
		#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15, Ryan+17
			#if(FULL_ENTROPY)
			pi[ENTRE] = 1.0 / (GAMMAE - 1.) * log(pe_new * pow(pi[RHO], -GAMMAE));
			pi[ENTRI] = 1.0 / (GAMMA - 1.) * log(pi_new * pow(pi[RHO], -GAMMA));
			#else
			pi[ENTRE] = pe_new * pow(pi[RHO], -GAMMAE);
			pi[ENTRI] = pi_new * pow(pi[RHO], -GAMMA);
			#endif
		#elif(VARGAMMA)   // variable gamma: Sadowski+17, Chael+19
		Theta_e = pe_new / pi[RHO] * MU_E * MASS_RATIO;
		Theta_i = pi_new / pi[RHO] * MU_I;
			#if(FULL_ENTROPY_VARGAMMA)
			pi[ENTRE] = log(pow(Theta_e, 1.5) * pow(Theta_e + 0.4, 1.5) / pi[RHO]);
			pi[ENTRI] = log(pow(Theta_i, 1.5) * pow(Theta_i + 0.4, 1.5) / pi[RHO]);
			#else
			pi[ENTRE] = Theta_e * (Theta_e + 0.4) / pow(pi[RHO], 2. / 3.);
			pi[ENTRI] = Theta_i * (Theta_i + 0.4) / pow(pi[RHO], 2. / 3.);
			#endif	
		#endif	
	
		//Calculate total error in delta
		delta = ue / (ue + ui);
		delta_f = calc_delta(pi, bsq);   // initial Tel/Ttot (temperature ratio)
		error_1 = delta_f - delta;

		//Calculate gradient of error
		derror_dpe = (error_1 - error_0) / (offset*pe_new);

		//Apply correction to electron pressure based on error and error gradient
		pe_old = pe_new / (1.0 + offset);
		pe_new = fabs(pe_old - error_0 / derror_dpe);

		//Calculate relative error, and if smaller than NEWT_TOL stop iterating
		errx = fabs(pe_new - pe_old) / pe_old;
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	//If converged set new gas internal energy
	if (fabs(errx) <= NEWT_TOL) {
		//Calculate (final) ion pressure
		pi_new = p_tot - pe_new;
	}
	else { //Set ion and electron entropies to half of total pressure
		pe_new = pi_new = 0.5 * p_tot;
		fprintf(stderr, "set_2T_entropy failed to converge! \n");
	}

	//Calculate electron and ion entropy primitive variables
	#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15, Ryan+17
		#if(FULL_ENTROPY)
		pi[ENTRE] = 1.0 / (GAMMAE - 1.) * log(pe_new * pow(pi[RHO], -GAMMAE));
		pi[ENTRI] = 1.0 / (GAMMA - 1.) * log(pi_new * pow(pi[RHO], -GAMMA));
		#else
		pi[ENTRE] = pe_new * pow(pi[RHO], -GAMMAE);
		pi[ENTRI] = pi_new * pow(pi[RHO], -GAMMA);
		#endif
	#elif(VARGAMMA)   // variable gamma: Sadowski+17, Chael+19
	Theta_e = pe_new / pi[RHO] * MU_E * MASS_RATIO;
	Theta_i = pi_new / pi[RHO] * MU_I;
		#if(FULL_ENTROPY_VARGAMMA)
		pi[ENTRE] = log(pow(Theta_e, 1.5) * pow(Theta_e + 0.4, 1.5) / pi[RHO]);
		pi[ENTRI] = log(pow(Theta_i, 1.5) * pow(Theta_i + 0.4, 1.5) / pi[RHO]);
		#else
		pi[ENTRE] = Theta_e * (Theta_e + 0.4) / pow(pi[RHO], 2. / 3.);
		pi[ENTRI] = Theta_i * (Theta_i + 0.4) / pow(pi[RHO], 2. / 3.);
		#endif	
	#endif	
	
	//Set gas internal energy based on total pressure and effective adiabatic index
	gamg = calc_gamma_gas_prim(pi);
	pi[UU] = p_tot / (gamg - 1.0);
}
#endif

void init_rad_pres(double pi[NPR]) {
	double T_old, T_new, ptot, pgas, prad, arad, dPdT, errx;
	int keep_iterating=1, i, n_iter=0;
	#if(!CALC_MDOT)
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;	
	#else
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	//Calculate old pressure
	#if(TWO_T)
	arad = (ARAD / energy_density_scale) * pow(MU_E * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.);
	#else
	arad = (ARAD / energy_density_scale) * pow(MU_G * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.);
	#endif

	#if(HIGH_MDOT)
	if (read_M1) {
		T_old = (GAMMA - 1.) * pi[UU] / pi[RHO];
		T_new = T_old;
		ptot = (GAMMA - 1.) * pi[UU];
	}
	else {
		T_old = (4. / 3. - 1.) * pi[UU] / pi[RHO];
		T_new = T_old;
		ptot = (4. / 3. - 1.) * pi[UU];
	}
	while (keep_iterating) {
		//Calculate gradient dPdT
		dPdT = pi[RHO] + 4. / 3.*arad*pow(T_new, 3.);

		/* Make the newton step: */
		T_old = T_new;
		T_new = T_old - ((pi[RHO] * T_old+1./3.*arad*pow(T_old,4.))-ptot) / dPdT;

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		errx = fabs(T_new-T_old)/T_old;

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER*5 - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	} 

	pgas = pi[RHO] * T_new;
	pi[UU] = pgas / (GAMMA - 1.); //note that based on EOS energy should be divided between electrons and ions
	pi[UU_RAD] = arad * pow(T_new, 4.);
	#else
	pi[UU_RAD] = pi[UU]*0.001;
	#endif

	//Set photon number based on Boltzman distribution
	#if(P_NUM)
	T_new = pow(pi[UU_RAD] * energy_density_scale / ARAD, 0.25);
	pi[PHOTON] = pi[UU_RAD] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * T_new);
	#endif

	pi[U1_RAD] = pi[U1];
	pi[U2_RAD] = pi[U2];
	pi[U3_RAD] = pi[U3];
}

#if(RESISTIVE)
void set_E_init(double p[NPR], struct of_geom geom) {
	int i1, j1, k1, l1, i, j, k, n;
	double alpha, sqrtgamma, B_guess[3], B_D[3], vd_guess[3], gamma;
	struct of_state state;
	
	get_state(p, &geom, &state);
	alpha = 1.0 / sqrt(-geom.gcon[0][0]);
	sqrtgamma = geom.g / alpha; //determinant for spatial part of metric
	gamma = alpha * state.ucon[0];
	vd_guess[0] = state.ucov[1] / gamma;
	vd_guess[1] = state.ucov[2] / gamma;
	vd_guess[2] = state.ucov[3] / gamma;
	B_guess[0] = alpha * p[B1];
	B_guess[1] = alpha * p[B2];
	B_guess[2] = alpha * p[B3];

	lower_3(B_guess, geom.gcov, B_D);

	for (i1 = 0; i1 < 3; i1++) {
		p[E1 + i1] = 0.;
		for (j1 = 0; j1 < 3; j1++)for (k1 = 0; k1 < 3; k1++) {
			if ((j1 == k1) || (j1 == i1) || (k1 == i1)) continue;
			p[E1 + i1] = p[E1 + i1] - (1.0 / geom.g * lvc3u(i1, j1, k1) * vd_guess[j1] * B_D[k1]);
		}
	}
}
#endif