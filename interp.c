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

/* performs the slope-limiting for the numerical flux calculation */

double slope_lim(double y1,double y2,double y3) 
{

	double Dqm, Dqp, Dqc, s;

	/* woodward, or monotonized central, slope limiter */
	if (lim == MC) {
		Dqm = 2.0*(y2 - y1);
		Dqp = 2.0*(y3 - y2);
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
	/* van leer slope limiter */
	else if (lim == VANL) {
		Dqm = (y2 - y1);
		Dqp = (y3 - y2);
		s = Dqm*Dqp;
		if (s <= 0.) return 0.;
		else
			return(2.*s / (Dqm + Dqp));
	}

	/* minmod slope limiter (crude but robust) */
	else if (lim == MINM) {
		Dqm = (y2 - y1);
		Dqp = (y3 - y2);
		s = Dqm*Dqp;
		if (s <= 0.) return 0.;
		else if (fabs(Dqm) < fabs(Dqp)) return Dqm;
		else return Dqp;
	}

	fprintf(stderr, "unknown slope limiter\n");
	exit(10);

	return(0.);
}

double B1_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double a0, a1, a2, a3, a11, a12, a13, b12, c13;
	double a23, a123, a113, a112, c123, b123;
	double B1p, B1m, d2B1p, d2B1m, d3B1p, d3B1m, d1B2p, d1B2m, d1B3p, d1B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	if (n_rec6 == 10){
		B1p = 0.25*(b1_5 + b1_8 + b1_6 + b1_7);
		B1m = 0.25*(b1_1 + b1_2 + b1_3 + b1_4);
	}
	else{
		B1p = pb[nl[n]][index_3D(n, i + D1, j, z)][1];
		B1m = pb[nl[n]][index_3D(n, i, j, z)][1];
	}

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i + D1, j - D2, z)][1], pb[nl[n]][index_3D(n, i + D1, j, z)][1], pb[nl[n]][index_3D(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z)][1], pb[nl[n]][index_3D(n, i, j, z)][1], pb[nl[n]][index_3D(n, i, j + D2, z)][1]);

	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i + D1, j, z - D3)][1], pb[nl[n]][index_3D(n, i + D1, j, z)][1], pb[nl[n]][index_3D(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j, z - D3)][1], pb[nl[n]][index_3D(n, i, j, z)][1], pb[nl[n]][index_3D(n, i, j, z + D3)][1]);

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j + D2, z)][2], pb[nl[n]][index_3D(n, i, j + D2, z)][2], pb[nl[n]][index_3D(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z)][2], pb[nl[n]][index_3D(n, i, j, z)][2], pb[nl[n]][index_3D(n, i + D1, j, z)][2]);

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z + D3)][3], pb[nl[n]][index_3D(n, i, j, z + D3)][3], pb[nl[n]][index_3D(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z)][3], pb[nl[n]][index_3D(n, i, j, z)][3], pb[nl[n]][index_3D(n, i + D1, j, z)][3]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	a113 = -b123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][1] * dx[nl[n]][1] * dx[nl[n]][3];
	a112 = -c123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][1] * dx[nl[n]][1] * dx[nl[n]][2];
	a1 = B1p - B1m;
	a2 = 0.5*(d2B1p + d2B1m) + c123 * dx[nl[n]][1] / (16. * dx[nl[n]][3]);
	a3 = 0.5*(d3B1p + d3B1m) + b123 * dx[nl[n]][1] / (16. * dx[nl[n]][2]);
	a23 = 0.5*(d23B1p + d23B1m);
	a12 = d2B1p - d2B1m;
	b12 = d1B2p - d1B2m;
	a13 = d3B1p - d3B1m;
	c13 = d1B3p - d1B3m;
	a11 = -0.5*(b12 / dx[nl[n]][2] + c13 / dx[nl[n]][3])*dx[nl[n]][1];
	a0 = 0.5*(B1p + B1m) - a11 / 4.;

	double var = a0 + a1*offset_1 + a2*offset_2 + a3*offset_3 + a11*offset_1*offset_1 + a12 * offset_1*offset_2 + a13*offset_1*offset_3 +
		a23*offset_2*offset_3 + a123*offset_1*offset_2*offset_3 + a113*offset_1*offset_1*offset_3 + a112*offset_1*offset_1*offset_2;
	return var;
}


double B2_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double b0, b1, b2, b3, b12, b13, b22, b23, a123, b123, c123, b223, b122;
	double B2p, B2m, a12, c23, d1B2p, d1B2m, d3B2p, d3B2m, d2B1p, d2B1m, d2B3p, d2B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j + D2, z)][2], pb[nl[n]][index_3D(n, i, j + D2, z)][2], pb[nl[n]][index_3D(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z)][2], pb[nl[n]][index_3D(n, i, j, z)][2], pb[nl[n]][index_3D(n, i + D1, j, z)][2]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j + D2, z - D3)][2], pb[nl[n]][index_3D(n, i, j + D2, z)][2], pb[nl[n]][index_3D(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j, z - D3)][2], pb[nl[n]][index_3D(n, i, j, z)][2], pb[nl[n]][index_3D(n, i, j, z + D3)][2]);

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i + D1, j - D2, z)][1], pb[nl[n]][index_3D(n, i + D1, j, z)][1], pb[nl[n]][index_3D(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z)][1], pb[nl[n]][index_3D(n, i, j, z)][1], pb[nl[n]][index_3D(n, i, j + D2, z)][1]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z + D3)][3], pb[nl[n]][index_3D(n, i, j, z + D3)][3], pb[nl[n]][index_3D(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z)][3], pb[nl[n]][index_3D(n, i, j, z)][3], pb[nl[n]][index_3D(n, i, j + D2, z)][3]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	if (n_rec6 == 1000000000){
		B2p = 0.25*(b2_3 + b2_8 + b2_4 + b2_7);
		B2m = 0.25*(b2_1 + b2_6 + b2_2 + b2_5);
	}
	else{
		B2p = pb[nl[n]][index_3D(n, i, j + D2, z)][2];
		B2m = pb[nl[n]][index_3D(n, i, j, z)][2];
	}
	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	b223 = -a123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][2] * dx[nl[n]][2] * dx[nl[n]][3];
	b122 = -c123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][2];
	b1 = 0.5*(d1B2p + d1B2m) + c123 * dx[nl[n]][2] / (16. * dx[nl[n]][3]);
	b2 = B2p - B2m;
	b3 = 0.5*(d3B2p + d3B2m) + a123 * dx[nl[n]][2] / (16. * dx[nl[n]][1]);
	b13 = 0.5*(d13B2p + d13B2m);
	b12 = d1B2p - d1B2m;
	b23 = d3B2p - d3B2m;
	a12 = d2B1p - d2B1m;
	c23 = d2B3p - d2B3m;
	b22 = -0.5*(a12 / dx[nl[n]][1] + c23 / dx[nl[n]][3])*dx[nl[n]][2];
	b0 = 0.5*(B2p + B2m) - b22 / 4.;
	double var = b0 + b1*offset_1 + b2*offset_2 + b3*offset_3 + b12*offset_1*offset_2 + b22*offset_2*offset_2 + b23*offset_2*offset_3 +
		b13*offset_1*offset_3 + b223*offset_2*offset_2*offset_3 + b123*offset_1*offset_2*offset_3 + b122*offset_1*offset_2*offset_2;
	return var;
}

double B3_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double c0, c1, c2, c3, c12, c13, c23, c33, a13, b23, a123, b123, c123, c233, c133;
	double B3p, B3m, d1B3p, d1B3m, d2B3p, d2B3m, d3B1p, d3B1m, d3B2p, d3B2m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z + D3)][3], pb[nl[n]][index_3D(n, i, j, z + D3)][3], pb[nl[n]][index_3D(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i - D1, j, z)][3], pb[nl[n]][index_3D(n, i, j, z)][3], pb[nl[n]][index_3D(n, i + D1, j, z)][3]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z + D3)][3], pb[nl[n]][index_3D(n, i, j, z + D3)][3], pb[nl[n]][index_3D(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j - D2, z)][3], pb[nl[n]][index_3D(n, i, j, z)][3], pb[nl[n]][index_3D(n, i, j + D2, z)][3]);

	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i + D1, j, z - D3)][1], pb[nl[n]][index_3D(n, i + D1, j, z)][1], pb[nl[n]][index_3D(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j, z - D3)][1], pb[nl[n]][index_3D(n, i, j, z)][1], pb[nl[n]][index_3D(n, i, j, z + D3)][1]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j + D2, z - D3)][2], pb[nl[n]][index_3D(n, i, j + D2, z)][2], pb[nl[n]][index_3D(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[nl[n]][index_3D(n, i, j, z - D3)][2], pb[nl[n]][index_3D(n, i, j, z)][2], pb[nl[n]][index_3D(n, i, j, z + D3)][2]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	if (n_rec6 == 1000000000){
		B3p = 0.25*(b3_2 + b3_4 + b3_6 + b3_8);
		B3m = 0.25*(b3_1 + b3_3 + b3_5 + b3_7);
	}
	else{
		B3p = pb[nl[n]][index_3D(n, i, j, z + D3)][3];
		B3m = pb[nl[n]][index_3D(n, i, j, z)][3];
	}

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	c233 = -a123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][2] * dx[nl[n]][3] * dx[nl[n]][3];;
	c133 = -b123 / 4. / (dx[nl[n]][1] * dx[nl[n]][2] * dx[nl[n]][3])*dx[nl[n]][1] * dx[nl[n]][3] * dx[nl[n]][3];;
	c1 = 0.5*(d1B3p + d1B3m) + b123 * dx[nl[n]][3] / (16.*dx[nl[n]][2]);
	c2 = 0.5*(d2B3p + d2B3m) + a123 * dx[nl[n]][3] / (16.*dx[nl[n]][1]);
	c3 = B3p - B3m;
	c12 = 0.5*(d12B3p + d12B3m);
	c13 = d1B3p - d1B3m;
	c23 = d2B3p - d2B3m;
	a13 = d3B1p - d3B1m;
	b23 = d3B2p - d3B2m;
	c33 = -0.5*(a13 / dx[nl[n]][1] + b23 / dx[nl[n]][2])*dx[nl[n]][3];
	c0 = 0.5*(B3p + B3m) - c33 / 4.;

	double var = c0 + c1*offset_1 + c2*offset_2 + c3*offset_3 + c13*offset_1*offset_3 + c23*offset_2*offset_3 + c33*offset_3*offset_3 +
		c12*offset_1*offset_2 + c233*offset_2*offset_3*offset_3 + c133*offset_1*offset_3*offset_3 + c123*offset_1*offset_2*offset_3;
	return var;
}

//Averages grid in blocks near pole in case of internal derefinement
void average_grid(void){
	int n, i, j, z, k, u;
	int zsize = 1, zlevel = 0;
	double temp[NPR];

	//Average primitive and staggered grid variables
	#if(N_LEVELS_1D_INT>0 && D3>0)
	for (n = 0; n < n_active; n++) {
		#if(N_GPU>1)
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		#pragma omp parallel for schedule(dynamic,1) private(i, j, z, k, temp, zsize, zlevel, u)
		for (i = N1_GPU_offset[n_ord[n]]; i <= N1_GPU_offset[n_ord[n]] + BS_1; i++)for (j = N2_GPU_offset[n_ord[n]]; j <= N2_GPU_offset[n_ord[n]] + BS_2; j++) {
			zlevel = 0;
			if ((block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3) && j < N2_GPU_offset[n_ord[n]] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n_ord[n]]) + D2))) / log(2.)), N_LEVELS_1D_INT);
			if ((block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3) && j >= N2_GPU_offset[n_ord[n]] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n_ord[n]], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
			zsize = (int)(0.001+pow(2.0, (double)zlevel));
			for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3; z += zsize) {
				PLOOP temp[k] = 0.0;
				for (u = 0; u < zsize; u++) PLOOP temp[k] += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][k] / ((double)zsize);
				for (u = 0; u < zsize; u++) PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][k] = temp[k];

				#if(STAGGERED)
				temp[1] = 0.0;
				for (u = 0; u < zsize; u++) temp[1] += (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][1]) / ((double)zsize);
				for (u = 0; u < zsize; u++) ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][1] = temp[1];

				temp[2] = 0.0;
				for (u = 0; u < zsize; u++) temp[2] += (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + (j >= (N2_GPU_offset[n_ord[n]] + BS_2 / 2)), z + u)][2] ) / ((double)zsize);
				for (u = 0; u < zsize; u++) ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + (j >= (N2_GPU_offset[n_ord[n]] + BS_2 / 2)), z + u)][2] = temp[2];

				temp[3] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE3] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + zsize)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + zsize)][FACE3]) / (2.0);
				for (u = 1; u < zsize; u++)ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][3] = temp[3] / gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + zsize / 2)][FACE3];
				#endif
			}
		}

		#if(GPU_ENABLED || GPU_DEBUG )
		#pragma omp parallel private(i, j, z, k)
		{
			#pragma omp for collapse(2) schedule(dynamic)
			ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, N1_GPU_offset[n_ord[n]] + BS_1 - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G) {
				for (k = 0; k < NPR; k++) {
					p_1[nl[n_ord[n]]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]]) + (i - N1_GPU_offset[n_ord[n]] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n_ord[n]] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n_ord[n]] + N3G)] = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];
				}
				#if(STAGGERED)
				for (k = 1; k < NDIM; k++) {
					ps_1[nl[n_ord[n]]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]]) + (i - N1_GPU_offset[n_ord[n]] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n_ord[n]] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n_ord[n]] + N3G)] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];
				}
				#endif
			}
		}
		cudaMemcpyAsync(Bufferp_1[nl[n_ord[n]]], p_1[nl[n_ord[n]]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]]) * sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n_ord[n]]]);
		#if(STAGGERED)
		cudaMemcpyAsync(Bufferps_1[nl[n_ord[n]]], ps_1[nl[n_ord[n]]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n_ord[n]]]) * sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n_ord[n]]]);
		#endif
		#endif
	}
	#endif
}

//Prolongs grid near pole: This is necessary for AMR in combination with internal derefinement
void prolong_grid(void){
	int n, i, j, z, zs, k, u, u2;
	int zsize = 1, zlevel = 0;
	double temp[NDIM];
	double b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8;
	double b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8;
	double b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	//Store staggered grid variables in temporary array
	for (n = 0; n < n_active; n++){
		#pragma omp parallel private(i, j, z)
		{
			#pragma omp for collapse(3) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)
			ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, N1_GPU_offset[n_ord[n]] + BS_1 + N1G - D1, -N2G + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + N2G - D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + N3G - D3) {
				psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE1];
				psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE2];
				psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE3];
			}
		}
	}
	for (n = 0; n < n_active; n++){
		#pragma omp parallel for collapse(2) schedule(static, (BS_1*2*N1G)*(BS_2*2*N2G)/nthreads) private(i, j, z, zs, k, temp, zsize, zlevel, u, u2, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8,b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8)
		for (i = N1_GPU_offset[n_ord[n]]; i < N1_GPU_offset[n_ord[n]] + BS_1; i++)for (j = N2_GPU_offset[n_ord[n]]; j <= N2_GPU_offset[n_ord[n]] + BS_2; j++){
			zlevel = 0;
			if ((block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3) && j < N2_GPU_offset[n_ord[n]] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n_ord[n]]) + D2))) / log(2.)), N_LEVELS_1D_INT);
			if ((block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3) && j >= N2_GPU_offset[n_ord[n]] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n_ord[n]], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
			zsize =(int)(0.001+pow(2.0, (double)zlevel));
			if (zlevel>0){
				for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3; z += zsize){
					for (zs = zsize; zs > 1; zs/=2) {
						for (u = zs / 2; u < zsize; u += zs) {
							b1_1 = b1_2 = b1_3 = b1_4 = 0.;
							b1_5 = b1_6 = b1_7 = b1_8 = 0.;
							b2_1 = b2_2 = b2_5 = b2_6 = 0.;
							b2_3 = b2_4 = b2_7 = b2_8 = 0.;

							for (u2 = u - zs/2; u2 < u; u2++) {
								//Negative x1
								b1_1 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][1] * 2.0;
								b1_3 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][1] * 2.0;

								//Positive x1
								b1_5 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z + u2)][1] * 2.0;
								b1_7 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z + u2)][1] * 2.0;
								
								//Negative x2
								b2_1 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][2] * 2.0;
								b2_5 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][2] * 2.0;

								//Positive x2
								b2_3 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z + u2)][2] * 2.0;
								b2_7 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z + u2)][2] * 2.0;
							}

							for (u2 = u; u2 < u + zs / 2; u2++) {
								//Negative x1
								b1_2 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][1] * 2.0;
								b1_4 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][1] * 2.0;

								//Positive x1
								b1_6 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z + u2)][1] * 2.0;
								b1_8 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z + u2)][1] * 2.0;

								//Negative x2
								b2_2 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][2] * 2.0;
								b2_6 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u2)][2] * 2.0;

								//Positive x2
								b2_4 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z + u2)][2] * 2.0;
								b2_8 += psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z + u2)][2] * 2.0;
							}

							//Negative x3
							b3_1 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u - zs / 2))][3];
							b3_3 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u - zs / 2))][3];
							b3_5 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u - zs / 2))][3];
							b3_7 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u - zs / 2))][3];

							//Positive x3
							b3_2 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u + zs / 2))][3];
							b3_4 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u + zs / 2))][3];
							b3_6 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u + zs / 2))][3];
							b3_8 = psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + (u + zs / 2))][3];

							psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][3] = B3_prolong(n_ord[n], i, j, z, 0, 0, 0.0, psh, b1_1, b1_2, b1_3, b1_4, b1_5, b1_6, b1_7, b1_8,
								b2_1, b2_2, b2_3, b2_4, b2_5, b2_6, b2_7, b2_8, b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, 1, 1, 1, 1, 1, 1000000000);
							ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][3] = 1. / gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + u)][FACE3] * psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + u)][3];
						}
					}
				}
			}
		}
	}
	#endif
}