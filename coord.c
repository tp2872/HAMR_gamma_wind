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
#include "decs_MPI.h"

/** 
 *
 * this file contains all the coordinate dependent
 * parts of the code, except the initial and boundary
 * conditions 
 *
 **/
/***************************************************************************/
/***************************************************************************
coord():
-------
-- given the indices i,j and location in the cell, return with
the values of X1,X2 there;
-- the locations are defined by :
-----------------------
|                     |
|                     |
|FACE1   CENT         |
|                     |
|CORN    FACE2        |
----------------------
***************************************************************************/
void coord(int n, int i, int j, int z, int loc, double * restrict X)
{
	X[0] = 0.0;
	int j_local = j;
	if (j < 0) j_local = -j - 1;
	if (j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL2])) j_local = 2 * N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j;
	if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && loc == FACE2) j_local = j;
	if (loc == FACE1) {
		X[1] = startx[1] + i*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else if (loc == FACE2) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + j_local*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else if (loc == FACE3) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + z*dx[nl[n]][3];
	}
	else if (loc == CENT) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else {
		X[1] = startx[1] + i*dx[nl[n]][1];
		X[2] = startx[2] + j_local*dx[nl[n]][2];
		X[3] = startx[3] + z*dx[nl[n]][3];
	}

	if (j < 0){
		X[2] = X[2] + 1;
		X[2] = -X[2];
		X[2] = X[2] - 1;
	}
	if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && loc == FACE2){
	}
	else if (j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL])){
		X[2] = X[2] + 1;
		X[2] = 4. - X[2];
		X[2] = X[2] - 1;
	}

	return;
}

/* should return boyer-lindquist coordinte of point */
void bl_coord(double * restrict X, double * restrict r, double * restrict th, double * restrict phi)
{
	double V[4];

	#if(!DOCYLINDRIFYCOORDS)
	vofx_matthewcoords(X,V);
	#else
	vofx_cylindrified(X, vofx_matthewcoords, V);
	#endif

	// avoid singularity at polar axis
	#if(COORDSINGFIX)
	if (fabs(V[2])<SINGSMALL){
		if (V[2] >= 0.0) V[2] = SINGSMALL;
		if (V[2]<0.0)  V[2] = -SINGSMALL;
	}
	if (fabs(M_PI - V[2]) <SINGSMALL){
		if (V[2] >= M_PI) V[2] = M_PI + SINGSMALL;
		if (V[2]<M_PI)  V[2] = M_PI -  SINGSMALL;
	}
	#endif

	*r = V[1];
	*th = V[2];
	*phi = V[3];
	return ;
}

void vofx_matthewcoords(double *X, double *V){
	V[0] = X[0];
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	if (X[1] < Xtrans){
		V[1] = exp(pow(X[1], RADEXP)) + RB;
	}
	else if (X[1] >= Xtrans && X[1]<1.01*Xtrans){
		V[1] = 10.*(X[1] / Xtrans - 1.)*((X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS) +
			(1. - 10.*(X[1] / Xtrans - 1.))*(exp(pow(X[1], RADEXP)) + RB);
	}
	else{
		V[1] = (X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS;
	}
	double A1 = 1. / (1. + pow(CHARLIE*(log(V[1]) / log(10.)), DELTA));
	double A2 = BRAVO*(log(V[1]) / log(10.)) + TANGO;
	double A3 = pow(0.5, 1. - A2);
	double sign = 1.;
	double X_2 =(X[2]+1.0)/2.0;
	double Xc = sqrt(pow(X_2, 2.));

	if (X_2 < 0.0){
		sign = -1.;
	}
	if (X_2 > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X_2 >= 0.5){
		Xc = 1. - Xc;
		V[2] = M_PI - sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	else{
		V[2] = sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	V[3] = X[3];
}

/* some grid location, dxs */
void set_points(int n)
{
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	if(Rout<=RTRANS){
		dx[nl[n]][1] = (pow(log(Rout - RB), 1. / RADEXP) - pow(log(Rin - RB), 1. / RADEXP)) / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
	}
	else{
		dx[nl[n]][1] = ((Rout - RTRANS + Xtrans *RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP)) / (RADEXP*exp(pow(Xtrans, RADEXP))*
			pow(Xtrans, -1. + RADEXP)) - pow(log(Rin), 1. / RADEXP)) / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
	}
	dx[nl[n]][2] = 2.*fractheta / (double)(N2) / (double)(pow(1 + REF_2, block[n][AMR_LEVEL2]));
	dx[nl[n]][3] = 2.*M_PI / (double)(N3) / (double)(pow(1 + REF_3, block[n][AMR_LEVEL3]));
}

void set_gridparam(void) {
	a = BH_SPIN;
	Rin = 0.9*(1. + sqrt(1. - a * a));
	Rout = 100000.;
	lim = MC;
	failed = 0;
	cour = COUR;
	if (dt > 1e-5) dt = dt;
	else dt = 1.e-4;
	R0 = 0.0;
	gam = GAMMA;

	if (N2 != 1) {
		//2D problem, use full pi-wedge in theta
		fractheta = 1.0 - 2.0 / ((double)N2)*(TRANS_BOUND == 1);
	}
	else {
		//1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
		fractheta = 1.e-2;
	}

	startx[1] = pow(log(Rin - RB), 1. / RADEXP);
	startx[2] = -1. + 1.*(1. - fractheta);
	startx[3] = 0.;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
//  CYLINDRIFICATION
//
//////////////////////////////////////////////////////////////////////////////////////////
//Adjusts V[2]=theta so that a few innermost cells around the pole
//become cylindrical
//ASSUMES: poles are at
//            X[2] = -1 and +1, which correspond to
//            V[2] = 0 and pi
void vofx_cylindrified(double *Xin, void(*vofx)(double*, double*), double *Vout)
{
	double npiovertwos;
	double X[NDIM], V[NDIM];
	double Vin[NDIM];
	double X0[NDIM], V0[NDIM];
	double Xtr[NDIM], Vtr[NDIM];
	double f1, f2, dftr;
	double sinth, th;
	int j, ismirrored;

	vofx(Xin, Vin);

	// BRING INPUT TO 1ST QUADRANT:  X[2] \in [-1 and 0]
	to1stquadrant(Xin, X, &ismirrored);
	vofx(X, V);

	//initialize X0: cylindrify region
	//X[1] < X0[1] && X[2] < X0[2] (value of X0[3] not used)
	X0[0] = Xin[0];
	/*disk 150^3 Rout 100 Rg-->100^3=25 Rg*/
	X0[1] = pow(log(38.*(double)N3 / 250.0 - RB), 1. / RADEXP);
	X0[2] = -1. + 1. / ((double)(N2));
	X0[3] = 0.;
	/*3D jet Rout 10000 Rg 1024x400x100*/
	/*X0[1] = pow(log(600. - RB), 1. / RADEXP);
	X0[2] = -1. + 3. / (double)N2;
	X0[3] = 0.;*/
	vofx(X0, V0);

	//{0, roughly midpoint between grid origin and x10, -1, 0}
	DLOOPA Xtr[j] = X[j];
	//3D jet
	//Xtr[1] = pow(log(0.5*(exp(pow(X0[1], RADEXP) + RB) + exp(pow(startx[1], RADEXP) + RB))), 1. / RADEXP);   //always bound to be between startx[1] and X0[1]
	Xtr[1] = pow(log(0.5*(exp(pow(X0[1],RADEXP))+RB + exp(pow(startx[1],RADEXP))+RB)-RB),1./RADEXP);   //always bound to be between startx[1] and X0[1]
	vofx(Xtr, Vtr);

	f1 = func1(X0, X, vofx);
	f2 = func2(X0, X, vofx);
	dftr = func2(X0, Xtr, vofx) - func1(X0, Xtr, vofx);

	// Compute new theta
	sinth = maxs(V[1] * f1, V[1] * f2, Vtr[1] * fabs(dftr) + SMALL) / V[1];

	th = asin(sinth);

	//initialize Vout with the original values
	DLOOPA Vout[j] = Vin[j];

	//apply change in theta in the original quadrant
	if (0 == ismirrored) {
		Vout[2] = Vin[2] + (th - V[2]);
	}
	else {
		//if mirrrored, flip the sign
		Vout[2] = Vin[2] - (th - V[2]);
	}
}
//smooth step function:
// Ftr = 0 if x < 0, Ftr = 1 if x > 1 and smoothly interps. in btw.
double Ftr(double x)
{
	double res;

	if (x <= 0.) {
		res = 0.;
	}
	else if (x >= 1) {
		res = 1.;
	}
	else {
		res = (64. + cos(5. * M_PI*x) + 70. * sin((M_PI*(-1. + 2. * x)) / 2.) + 5. * sin((3. * M_PI*(-1. + 2. * x)) / 2.)) / 128.;
	}

	return(res);
}

double Ftrgenlin(double x, double xa, double xb, double ya, double yb)
{
	double Ftr(double x);
	double res;

	res = (x*ya) / xa + (-((x*ya) / xa) + ((x - xb)*(1. - yb)) / (1. - xb) + yb)*Ftr((x - xa) / (-xa + xb));

	return(res);
}

//goes from ya to yb as x goes from xa to xb
double Ftrgen(double x, double xa, double xb, double ya, double yb)
{
	double Ftr(double x);
	double res;

	res = ya + (yb - ya)*Ftr((x - xa) / (xb - xa));

	return(res);
}

double Fangle(double x)
{
	double res;

	if (x <= -1.) {
		res = 0.;
	}
	else if (x >= 1.) {
		res = x;
	}
	else {
		res = (1. + x + (-140. * sin((M_PI*(1. + x)) / 2.) + (10. * sin((3. * M_PI*(1. + x)) / 2.)) / 3. + (2. * sin((5. * M_PI*(1. + x)) / 2.)) / 5.) / (64.*M_PI)) / 2.;
	}

	return(res);

}

double limlin(double x, double x0, double dx, double y0)
{
	double Fangle(double x);
	return(y0 - dx * Fangle(-(x - x0) / dx));
}

double minlin(double x, double x0, double dx, double y0)
{
	double Fangle(double x);
	return(y0 + dx * Fangle((x - x0) / dx));
}

double mins(double f1, double f2, double df)
{
	double limlin(double x, double x0, double dx, double y0);
	return(limlin(f1, f2, df, f2));
}

double maxs(double f1, double f2, double df)
{
	double mins(double f1, double f2, double df);
	return(-mins(-f1, -f2, df));
}

//=mins if dir < 0
//=maxs if dir >= 0
double minmaxs(double f1, double f2, double df, double dir)
{
	double mins(double f1, double f2, double df);
	double maxs(double f1, double f2, double df);
	if (dir >= 0) {
		return(maxs(f1, f2, df));
	}

	return(mins(f1, f2, df));
}

//Converts copies Xin to Xout and converts
//but sets Xout[2] to lie in the 1st quadrant, i.e. Xout[2] \in [-1,0])
//if the point had to be mirrored
void to1stquadrant(double *Xin, double *Xout, int *ismirrored)
{
	double ntimes;
	int j;

	DLOOPA Xout[j] = Xin[j];

	//bring the angle variables to -2..2 (for X) and -2pi..2pi (for V)
	ntimes = floor((Xin[2] + 2.0) / 4.0);
	//this forces -2 < Xout[2] < 2
	Xout[2] -= 4. * ntimes;

	*ismirrored = 0;

	if (Xout[2] > 0.) {
		Xout[2] = -Xout[2];
		*ismirrored = 1 - *ismirrored;
	}

	//now force -1 < Xout[2] < 0
	if (Xout[2] < -1.) {
		Xout[2] = -2. - Xout[2];
		*ismirrored = 1 - *ismirrored;
	}
}

double sinth0(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V0[NDIM];
	double Vc0[NDIM];
	double Xc0[NDIM];
	int j;

	//X1 = {0, X[1], X0[1], 0}
	DLOOPA Xc0[j] = X[j];
	Xc0[2] = X0[2];

	vofx(Xc0, Vc0);
	vofx(X0, V0);


	return(V0[1] * sin(V0[2]) / Vc0[1]);
}

double sinth1in(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double V0[NDIM];
	double V0c[NDIM];
	double X0c[NDIM];
	int j;

	//X1 = {0, X[1], X0[1], 0}
	DLOOPA X0c[j] = X0[j];
	X0c[2] = X[2];

	vofx(X, V);
	vofx(X0c, V0c);
	vofx(X0, V0);

	return(V0[1] * sin(V0c[2]) / V[1]);
}


double th2in(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double V0[NDIM];
	double Vc0[NDIM];
	double Xc0[NDIM];
	double Xcmid[NDIM];
	double Vcmid[NDIM];
	int j;
	double res;
	double th0;

	DLOOPA Xc0[j] = X[j];
	Xc0[2] = X0[2];
	vofx(Xc0, Vc0);

	DLOOPA Xcmid[j] = X[j];
	Xcmid[2] = 0.;
	vofx(Xcmid, Vcmid);

	vofx(X0, V0);
	vofx(X, V);

	th0 = asin(sinth0(X0, X, vofx));

	res = (V[2] - Vc0[2]) / (Vcmid[2] - Vc0[2]) * (Vcmid[2] - th0) + th0;

	return(res);
}

double func1(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];

	vofx(X, V);

	return(sin(V[2]));
}

double func2(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double Xca[NDIM];
	double func2;
	int j;
	double sth1in, sth2in, sth1inaxis, sth2inaxis;

	//{0, X[1], -1, 0}
	DLOOPA Xca[j] = X[j];
	Xca[2] = -1.;

	vofx(X, V);

	sth1in = sinth1in(X0, X, vofx);
	sth2in = sin(th2in(X0, X, vofx));

	sth1inaxis = sinth1in(X0, Xca, vofx);
	sth2inaxis = sin(th2in(X0, Xca, vofx));

	func2 = minmaxs(sth1in, sth2in, fabs(sth2inaxis - sth1inaxis) + SMALL, X[1] - X0[1]);

	return(func2);
}
