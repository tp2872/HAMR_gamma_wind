
/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/

#include "decs.h"
/* insert metric here */
void gcov_func(double *X, double gcovp[][NDIM])
{
	int i, j, k, l;
	double sth, cth, s2, rho2, sph, cph;
	double r, th, phi;
	double tfac, rfac, hfac, pfac;
	double gcov[NDIM][NDIM];
	double dxdxp[NDIM][NDIM], dxdr[NDIM][NDIM], drdx[NDIM][NDIM], dxdxt[NDIM][NDIM], dxtdx[NDIM][NDIM];
	double V[NDIM], Vp[NDIM];
	double T1, T2, P1, P2, A;
	double offset = 0.000000001;
	double tilt = TILT_ANGLE / 180.*M_PI;
	DLOOP gcov[j][k] = 0.;
#if(NSY)
	bl_coord(X, &r, &th, &phi);

	//compute Jacobian nt->t (dt/dnt)
	dxdxt[0][0] = 1.;
	dxdxt[0][1] = 0.;
	dxdxt[0][2] = 0.;
	dxdxt[0][3] = 0.;
	dxdxt[1][0] = 0.;
	dxdxt[1][1] = cos(tilt);
	dxdxt[1][2] = 0.;
	dxdxt[1][3] = -sin(tilt);
	dxdxt[2][0] = 0.;
	dxdxt[2][1] = 0.;
	dxdxt[2][2] = 1.;
	dxdxt[2][3] = 0.;
	dxdxt[3][0] = 0.;
	dxdxt[3][1] = sin(tilt);
	dxdxt[3][2] = 0.0;
	dxdxt[3][3] = cos(tilt);
	invert_matrix(dxdxt, dxtdx);

	//compute Jacobian x1,x2,x3 -> r,th,phi (dr/dx1)
	dxdxp_func(X, dxdxp);

	Vp[1] = r*sin(th)*cos(phi);
	Vp[2] = r*sin(th)*sin(phi);
	Vp[3] = r*cos(th);

	V[1] = Vp[1] * cos(-tilt) - Vp[3] * sin(-tilt);
	V[2] = Vp[2];
	V[3] = sin(-tilt)*Vp[1] + cos(-tilt)*Vp[3];
	Vp[1] = sqrt(V[1] * V[1] + V[2] * V[2] + V[3] * V[3]);
	Vp[2] = acos(V[3] / Vp[1]);
	Vp[3] = atan2(V[2], V[1]);
	if (Vp[2] < 0.0) Vp[2] *= -1;
	if (Vp[2] > M_PI) Vp[2] = M_PI - (Vp[2] - M_PI);

#if(COORDSINGFIX)
	if (fabs(Vp[2])<SINGSMALL){
		if (Vp[2] >= 0.0) Vp[2] = SINGSMALL;
		if (Vp[2]<0.0)  Vp[2] = -SINGSMALL;
	}
	if (fabs(M_PI - Vp[2]) <SINGSMALL){
		if (Vp[2] >= M_PI) Vp[2] = M_PI + SINGSMALL;
		if (Vp[2]<M_PI)  Vp[2] = M_PI - SINGSMALL;
	}
#endif
	//fprintf(stderr, "r: %f %f, th: %f %f, phi: %f %f \n", r,Vp[1], th,Vp[2], phi,Vp[3]);
	r = Vp[1];
	th = Vp[2];
	phi = Vp[3];

	cth = cos(th);
	sth = sin(th);

	s2 = sth*sth;
	rho2 = r*r + a*a*cth*cth;

	gcov[0][0] = (-1. + 2.*r / rho2);
	gcov[0][1] = (2.*r / rho2);
	gcov[0][3] = (-2.*a*r*s2 / rho2);

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2.*r / rho2);
	gcov[1][3] = (-a*s2*(1. + 2.*r / rho2));

	gcov[2][2] = rho2;

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r / rho2));
#else
	bl_coord(X, &r, &th, &phi);

	cth = cos(th);
	sth = sin(th);

	s2 = sth*sth;
	rho2 = r*r + a*a*cth*cth;

	//compute Jacobian x1,x2,x3 -> r,th,phi (dr/dx1)
	dxdxp_func(X, dxdxp);

	gcov[0][0] = (-1. + 2.*r / rho2);
	gcov[0][1] = (2.*r / rho2);
	gcov[0][3] = (-2.*a*r*s2 / rho2);

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2.*r / rho2);
	gcov[1][3] = (-a*s2*(1. + 2.*r / rho2));

	gcov[2][2] = rho2;

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r / rho2));
#endif

#if(NSY)
	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(Vp[2])*cos(Vp[3]);
	dxdr[1][2] = r*cos(Vp[2])*cos(Vp[3]);
	dxdr[1][3] = -r*sin(Vp[2])*sin(Vp[3]);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(Vp[2])*sin(Vp[3]);
	dxdr[2][2] = r*cos(Vp[2])*sin(Vp[3]);
	dxdr[2][3] = r*sin(Vp[2])*cos(Vp[3]);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(Vp[2]);
	dxdr[3][2] = -r*sin(Vp[2]);
	dxdr[3][3] = 0.;
	invert_matrix(dxdr, drdx);

	//convert from kerr schild to cartesian coordinates
	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			gcovp[i][j] = 0.;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					gcovp[i][j] += gcov[k][l] * drdx[k][i] * drdx[l][j];
				}
			}
		}
	}
	//convert from cartesian to tilted cartesian coordinates
	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			gcov[i][j] = 0.;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					gcov[i][j] += gcovp[k][l] * dxtdx[k][i] * dxtdx[l][j];
				}
			}
		}
	}

	//compute Jacobian r,th,phi->x,y,z (dx/dr)
	bl_coord(X, &r, &th, &phi);

	dxdr[0][0] = 1.;
	dxdr[0][1] = 0.;
	dxdr[0][2] = 0.;
	dxdr[0][3] = 0.;
	dxdr[1][0] = 0.;
	dxdr[1][1] = sin(th)*cos(phi);
	dxdr[1][2] = r*cos(th)*cos(phi);
	dxdr[1][3] = -r*sin(th)*sin(phi);
	dxdr[2][0] = 0.;
	dxdr[2][1] = sin(th)*sin(phi);
	dxdr[2][2] = r*cos(th)*sin(phi);
	dxdr[2][3] = r*sin(th)*cos(phi);
	dxdr[3][0] = 0.;
	dxdr[3][1] = cos(th);
	dxdr[3][2] = -r*sin(th);
	dxdr[3][3] = 0.;

	//convert back to tilted kerr-schild coordinates
	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			gcovp[i][j] = 0.;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					gcovp[i][j] += gcov[k][l] * dxdr[k][i] * dxdr[l][j];
				}
			}
		}
	}
	for (i = 0; i < NDIM; i++){
		for (j = 0; j < NDIM; j++){
			gcov[i][j] = gcovp[i][j];
		}
	}
#endif
	//convert to code coordinates
	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			gcovp[i][j] = 0.;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					gcovp[i][j] += gcov[k][l] * dxdxp[k][i] * dxdxp[l][j];
				}
			}
		}
	}
}

/* assumes gcov has been set first; returns determinant */
double gdet_func(double gcov[][NDIM]) 
{
  int i,j,k;
  int permute[NDIM]; 
  double gcovtmp[NDIM][NDIM];
  double detg;
  for( i = 0 ; i < NDIM*NDIM ; i++ ) {  gcovtmp[0][i] = gcov[0][i]; }
  if( LU_decompose( gcovtmp,  permute ) != 0  ) { 
    fprintf(stderr, "gdet_func(): singular matrix encountered! \n");
    fail(FAIL_METRIC);
  }
  detg = 1.;
  DLOOPA detg *= gcovtmp[j][j];
  return( sqrt(fabs(detg)) );

}

/* invert gcov to get gcon */
void gcon_func(double gcov[][NDIM], double gcon[][NDIM])
{
  invert_matrix( gcov, gcon );
}

/***************************************************************************/
/***************************************************************************
  conn_func():
  -----------

   -- this gives the connection coefficient
	\Gamma^{i}_{j,k} = conn[..][i][j][k]
   --  where i = {1,2,3,4} corresponds to {t,r,theta,phi}

***************************************************************************/

/* Sets the spatial discretization in numerical derivatives : */
#define EPS 1.e-5

/* NOTE: parameter hides global variable */
void conn_func(double *X, struct of_geom *geom, double conn[][NDIM][NDIM])
{
	int i,j,k,l ;
	double tmp[NDIM][NDIM][NDIM] ;
	double Xh[NDIM],Xl[NDIM] ;
	double gh[NDIM][NDIM] ;
	double gl[NDIM][NDIM] ;

	for(k=0;k<NDIM;k++) {
		for(l=0;l<NDIM;l++) Xh[l] = X[l] ;
		for(l=0;l<NDIM;l++) Xl[l] = X[l] ;
		Xh[k] += EPS ;
		Xl[k] -= EPS ;
		gcov_func(Xh,gh) ;
		gcov_func(Xl,gl) ;
		for(i=0;i<NDIM;i++)
		for(j=0;j<NDIM;j++) 
			conn[i][j][k] = (gh[i][j] - gl[i][j])/(Xh[k] - Xl[k]) ;
	}

	/* now rearrange to find \Gamma_{ijk} */
	for(i=0;i<NDIM;i++)
	for(j=0;j<NDIM;j++)
	for(k=0;k<NDIM;k++) 
		tmp[i][j][k] = 0.5*(conn[j][i][k] + conn[k][i][j] - conn[k][j][i]) ;

	/* finally, raise index */
	for(i=0;i<NDIM;i++)
	for(j=0;j<NDIM;j++)
	for(k=0;k<NDIM;k++)  {
		conn[i][j][k] = 0. ;
		for(l=0;l<NDIM;l++) conn[i][j][k] += geom->gcon[i][l]*tmp[l][j][k] ;
	}
	/* done! */
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
void lower(double * restrict ucon, struct of_geom * restrict geom, double * restrict ucov)
{
	int i, j;
	 #pragma ivdep
	for (i = 0; i < NDIM; i++){
		ucov[i] = 0.0;
	}
	for (j = 0; j < NDIM; j++){
		 #pragma ivdep
		for (i = 0; i < NDIM; i++){
			ucov[i] += geom->gcov[i][j] * ucon[j];
		}
	}
    return ;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
void raise(double * restrict ucov, struct of_geom * restrict geom, double * restrict ucon)
{
	int i, j;
	 #pragma ivdep
	for (i = 0; i < NDIM; i++){
		ucon[i] = 0.0;
	}
	for (j = 0; j < NDIM; j++){
		 #pragma ivdep
		for (i = 0; i < NDIM; i++){
			ucon[i] += geom->gcon[i][j] * ucov[j];
		}
	}
    return ;
}

/* NOTE: parameter hides global variable */
void dxdxp_func(double *X, double dxdxp[][NDIM])
{
	int i, j, k, l;
	double Xh[NDIM], Xl[NDIM];
	double Vh[NDIM], Vl[NDIM];

	for (k = 0; k<NDIM; k++) {
		for (l = 0; l<NDIM; l++) Xh[l] = X[l];
		for (l = 0; l<NDIM; l++) Xl[l] = X[l];
		Xh[k] += 0.00001;
		Xl[k] -= 0.00001;
		Vh[0] = Xh[0];
		Vl[0] = Xl[0];
		bl_coord(Xh, &Vh[1], &Vh[2], &Vh[3]);
		bl_coord(Xl, &Vl[1], &Vl[2], &Vl[3]);
		for (j = 0; j<NDIM; j++)
			dxdxp[j][k] = (Vh[j] - Vl[j]) / (Xh[k] - Xl[k]);
	}
}

/* load local geometry into structure geom */
void get_geometry(int n, int ii, int jj, int zz, int ff, struct of_geom * restrict geom)
{
	int i, j;
	for (i = 0; i < NDIM; i++){
		 #pragma ivdep
		for (j = 0; j < NDIM; j++){
			geom->gcon[i][j] = gcon[nl[n]][index_2D(n,ii, jj, zz)][ff][i][j];
			geom->gcov[i][j] = gcov[nl[n]][index_2D(n, ii, jj, zz)][ff][i][j];
		}
	}
	#if(GPU_DEBUG)
	geom->gcon[1][0] = geom->gcon[0][1];
	geom->gcov[1][0] = geom->gcov[0][1];
	geom->gcon[2][0] = geom->gcon[0][2];
	geom->gcov[2][0] = geom->gcov[0][2];
	geom->gcon[2][1] = geom->gcon[1][2];
	geom->gcov[2][1] = geom->gcov[1][2];
	geom->gcon[3][0] = geom->gcon[0][3];
	geom->gcov[3][0] = geom->gcov[0][3];
	geom->gcon[3][1] = geom->gcon[1][3];
	geom->gcov[3][1] = geom->gcov[1][3];
	geom->gcon[3][2] = geom->gcon[2][3];
	geom->gcov[3][2] = geom->gcov[2][3];
	#endif
	geom->g = gdet[nl[n]][index_2D(n, ii, jj, zz)][ff];
}

/* load local geometry into structure geom */
void get_trans(int n, int ii, int jj, int zz, int ff, struct of_trans * restrict trans)
{
	int i, j;
	for (i = 0; i < NDIM; i++){
		for (j = 0; j < NDIM; j++){
			trans->Mud[i][j] = Mud[nl[n]][index_2D(n, ii, jj, zz)][ff][i][j];
			trans->Mud_inv[i][j] = Mud_inv[nl[n]][index_2D(n, ii, jj, zz)][ff][i][j];
		}
	}
}

/*Load local geometry into structure geom for cases where the values are not stored in the memory 
such as during image output for MPI on the host node*/
void get_geometry_direct(int ii, int jj, int zz, int ff, struct of_geom *geom)
{
	int j, k;
	double X[NDIM];
	double gcov_local[NDIM][NDIM], gcon_local[NDIM][NDIM], gdet_local;
	coord(0, ii, jj,zz, ff, X);
	gcov_func(X, gcov_local);
	gcon_func(gcov_local, gcon_local);
	gdet_local = gdet_func(gcov_local);
	for (j = 0; j <= NDIM*NDIM - 1; j++){
		geom->gcon[0][j] = gcon_local[0][j];
		geom->gcov[0][j] = gcov_local[0][j];
	}
	geom->g = gdet_local;
}

#undef EPS

/* Boyer-Lindquist ("bl") metric functions */
void blgset(int n, int i, int j, struct of_geom *geom)
{
	double r, th,phi, X[NDIM];

	coord(n, i, j, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);

	if (th < 0) th *= -1.;
	if (th > M_PI) th = 2.*M_PI - th;

	geom->g = bl_gdet_func(r, th);
	bl_gcov_func(r, th, geom->gcov);
	bl_gcon_func(r, th, geom->gcon);
}

double bl_gdet_func(double r, double th)
{
	double a2, r2;

	a2 = a*a;
	r2 = r*r;
	return(
		r*r*fabs(sin(th))*(1. + 0.5*(a2 / r2)*(1. + cos(2.*th)))
		);
}

void kerr_gcov_func(double r, double th, double gcov[][NDIM])
{
	int j, k;
	double sth, cth, s2, a2, rho2, DD, mu;

	DLOOP gcov[j][k] = 0.;

	sth = fabs(sin(th));
	s2 = sth*sth;
	cth = cos(th);
	a2 = a*a;
	rho2 = r*r + a*a*cth*cth;	

	gcov[0][0] = (-1. + 2.*r / rho2);
	gcov[0][1] = (2.*r / rho2);
	gcov[0][3] = (-2.*a*r*s2 / rho2);

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2.*r / rho2);
	gcov[1][3] = (-a*s2*(1. + 2.*r / rho2));

	gcov[2][2] = rho2;

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r / rho2));
}

void bl_gcov_func(double r, double th, double gcov[][NDIM])
{
	int j, k;
	double sth, cth, s2, a2, r2, DD, mu;

	DLOOP gcov[j][k] = 0.;

	sth = fabs(sin(th));
	s2 = sth*sth;
	cth = cos(th);
	a2 = a*a;
	r2 = r*r;
	DD = 1. - 2. / r + a2 / r2;
	mu = 1. + a2*cth*cth / r2;

	gcov[0][0] = -(1. - 2. / (r*mu));
	gcov[0][3] = -2.*a*s2 / (r*mu);
	gcov[3][0] = gcov[0][3];
	gcov[1][1] = mu / DD;
	gcov[2][2] = r2*mu;
	gcov[3][3] = r2*sth*sth*(1. + a2 / r2 + 2.*a2*s2 / (r2*r*mu));

}

void bl_gcon_func(double r, double th, double gcon[][NDIM])
{
	int j, k;
	double sth, cth, a2, r2, r3, DD, mu;

	DLOOP gcon[j][k] = 0.;

	sth = sin(th);
	cth = cos(th);

	#if(COORDSINGFIX)
	if (fabs(sth) < SINGSMALL) {
		if (sth >= 0) sth = SINGSMALL;
		if (sth<0) sth = -SINGSMALL;
	}
	#endif

	a2 = a*a;
	r2 = r*r;
	r3 = r2*r;
	DD = 1. - 2. / r + a2 / r2;
	mu = 1. + a2*cth*cth / r2;

	gcon[0][0] = -1. - 2.*(1. + a2 / r2) / (r*DD*mu);
	gcon[0][3] = -2.*a / (r3*DD*mu);
	gcon[3][0] = gcon[0][3];
	gcon[1][1] = DD / mu;
	gcon[2][2] = 1. / (r2*mu);
	gcon[3][3] = (1. - 2. / (r*mu)) / (r2*sth*sth*DD);
}