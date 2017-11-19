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
		Dqm = 1.5*(y2 - y1);
		Dqp = 1.5*(y3 - y2);
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

double B1_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double a0, a1, a2, a3, a11, a12, a13, b12, c13;
	double a23, a123, a113, a112, c123, b123;
	double B1p, B1m, d2B1p, d2B1m, d3B1p, d3B1m, d1B2p, d1B2m, d1B3p, d1B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	B1p = pb[n][index_3D(n, i + D1, j, z)][1];
	B1m = pb[n][index_3D(n, i, j, z)][1];

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[n][index_3D(n, i + D1, j - D2, z)][1], pb[n][index_3D(n, i + D1, j, z)][1], pb[n][index_3D(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z)][1], pb[n][index_3D(n, i, j, z)][1], pb[n][index_3D(n, i, j + D2, z)][1]);

	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[n][index_3D(n, i + D1, j, z - D3)][1], pb[n][index_3D(n, i + D1, j, z)][1], pb[n][index_3D(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[n][index_3D(n, i, j, z - D3)][1], pb[n][index_3D(n, i, j, z)][1], pb[n][index_3D(n, i, j, z + D3)][1]);

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j + D2, z)][2], pb[n][index_3D(n, i, j + D2, z)][2], pb[n][index_3D(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z)][2], pb[n][index_3D(n, i, j, z)][2], pb[n][index_3D(n, i + D1, j, z)][2]);

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z + D3)][3], pb[n][index_3D(n, i, j, z + D3)][3], pb[n][index_3D(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z)][3], pb[n][index_3D(n, i, j, z)][3], pb[n][index_3D(n, i + D1, j, z)][3]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	a113 = -b123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][1] * dx[n][3];
	a112 = -c123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][1] * dx[n][2];
	a1 = B1p - B1m;
	a2 = 0.5*(d2B1p + d2B1m) + c123 * dx[n][1] / (16. * dx[n][3]);
	a3 = 0.5*(d3B1p + d3B1m) + b123 * dx[n][1] / (16. * dx[n][2]);
	a23 = 0.5*(d23B1p + d23B1m);
	a12 = d2B1p - d2B1m;
	b12 = d1B2p - d1B2m;
	a13 = d3B1p - d3B1m;
	c13 = d1B3p - d1B3m;
	a11 = -0.5*(b12 / dx[n][2] + c13 / dx[n][3])*dx[n][1];
	a0 = 0.5*(B1p + B1m) - a11 / 4.;

	double var = a0 + a1*offset_1 + a2*offset_2 + a3*offset_3 + a11*offset_1*offset_1 + a12 * offset_1*offset_2 + a13*offset_1*offset_3 +
		a23*offset_2*offset_3 + a123*offset_1*offset_2*offset_3 + a113*offset_1*offset_1*offset_3 + a112*offset_1*offset_1*offset_2;
	return var;
}


double B2_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double b0, b1, b2, b3, b12, b13, b22, b23, a123, b123, c123, b223, b122;
	double B2p, B2m, a12, c23, d1B2p, d1B2m, d3B2p, d3B2m, d2B1p, d2B1m, d2B3p, d2B3m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B2p = b2_7 + b2_8 - b2_3 - b2_4; //3,4,7,8
	d1B2m = b2_5 + b2_6 - b2_1 - b2_2; //1,2,5,6
	if (n_rec1 < 0) d1B2p = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j + D2, z)][2], pb[n][index_3D(n, i, j + D2, z)][2], pb[n][index_3D(n, i + D1, j + D2, z)][2]);
	if (n_rec3 < 0) d1B2m = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z)][2], pb[n][index_3D(n, i, j, z)][2], pb[n][index_3D(n, i + D1, j, z)][2]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[n][index_3D(n, i, j + D2, z - D3)][2], pb[n][index_3D(n, i, j + D2, z)][2], pb[n][index_3D(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[n][index_3D(n, i, j, z - D3)][2], pb[n][index_3D(n, i, j, z)][2], pb[n][index_3D(n, i, j, z + D3)][2]);

	d2B1p = b1_7 + b1_8 - b1_5 - b1_6; //5,6,7,8
	d2B1m = b1_3 + b1_4 - b1_2 - b1_1; //1,2,3,4
	if (n_rec4 < 0) d2B1p = 0.;// slope_lim(pb[n][index_3D(n, i + D1, j - D2, z)][1], pb[n][index_3D(n, i + D1, j, z)][1], pb[n][index_3D(n, i + D1, j + D2, z)][1]);
	if (n_rec2 < 0) d2B1m = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z)][1], pb[n][index_3D(n, i, j, z)][1], pb[n][index_3D(n, i, j + D2, z)][1]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z + D3)][3], pb[n][index_3D(n, i, j, z + D3)][3], pb[n][index_3D(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z)][3], pb[n][index_3D(n, i, j, z)][3], pb[n][index_3D(n, i, j + D2, z)][3]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	B2p = pb[n][index_3D(n, i, j + D2, z)][2];
	B2m = pb[n][index_3D(n, i, j, z)][2];

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	b223 = -a123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][2] * dx[n][2] * dx[n][3];
	b122 = -c123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][2] * dx[n][2];
	b1 = 0.5*(d1B2p + d1B2m) + c123 * dx[n][2] / (16. * dx[n][3]);
	b2 = B2p - B2m;
	b3 = 0.5*(d3B2p + d3B2m) + a123 * dx[n][2] / (16. * dx[n][1]);
	b13 = 0.5*(d13B2p + d13B2m);
	b12 = d1B2p - d1B2m;
	b23 = d3B2p - d3B2m;
	a12 = d2B1p - d2B1m;
	c23 = d2B3p - d2B3m;
	b22 = -0.5*(a12 / dx[n][1] + c23 / dx[n][3])*dx[n][2];
	b0 = 0.5*(B2p + B2m) - b22 / 4.;
	double var = b0 + b1*offset_1 + b2*offset_2 + b3*offset_3 + b12*offset_1*offset_2 + b22*offset_2*offset_2 + b23*offset_2*offset_3 +
		b13*offset_1*offset_3 + b223*offset_2*offset_2*offset_3 + b123*offset_1*offset_2*offset_3 + b122*offset_1*offset_2*offset_2;
	return var;
}

double B3_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6){
	double c0, c1, c2, c3, c12, c13, c23, c33, a13, b23, a123, b123, c123, c233, c133;
	double B3p, B3m, d1B3p, d1B3m, d2B3p, d2B3m, d3B1p, d3B1m, d3B2p, d3B2m;
	double d23B1p, d23B1m, d13B2p, d13B2m, d12B3p, d12B3m;

	d1B3p = b3_6 + b3_8 - b3_2 - b3_4; //2,4,6,8
	d1B3m = b3_5 + b3_7 - b3_1 - b3_3; //1,3,5,7
	if (n_rec6 < 0) d1B3p = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z + D3)][3], pb[n][index_3D(n, i, j, z + D3)][3], pb[n][index_3D(n, i + D1, j, z + D3)][3]);
	if (n_rec5 < 0) d1B3m = 0.;// slope_lim(pb[n][index_3D(n, i - D1, j, z)][3], pb[n][index_3D(n, i, j, z)][3], pb[n][index_3D(n, i + D1, j, z)][3]);

	d2B3p = b3_4 + b3_8 - b3_2 - b3_6; //2,4,6,8
	d2B3m = b3_3 + b3_7 - b3_1 - b3_5; //1,3,5,7
	if (n_rec6 < 0) d2B3p = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z + D3)][3], pb[n][index_3D(n, i, j, z + D3)][3], pb[n][index_3D(n, i, j + D2, z + D3)][3]);
	if (n_rec5 < 0) d2B3m = 0.;// slope_lim(pb[n][index_3D(n, i, j - D2, z)][3], pb[n][index_3D(n, i, j, z)][3], pb[n][index_3D(n, i, j + D2, z)][3]);

	d3B1p = b1_6 + b1_8 - b1_5 - b1_7; //5,6,7,8
	d3B1m = b1_2 + b1_4 - b1_1 - b1_3; //1,2,3,4
	if (n_rec4 < 0) d3B1p = 0.;// slope_lim(pb[n][index_3D(n, i + D1, j, z - D3)][1], pb[n][index_3D(n, i + D1, j, z)][1], pb[n][index_3D(n, i + D1, j, z + D3)][1]);
	if (n_rec2 < 0) d3B1m = 0.;// slope_lim(pb[n][index_3D(n, i, j, z - D3)][1], pb[n][index_3D(n, i, j, z)][1], pb[n][index_3D(n, i, j, z + D3)][1]);

	d3B2p = b2_4 + b2_8 - b2_3 - b2_7; //3,7,4,8
	d3B2m = b2_2 + b2_6 - b2_1 - b2_5; //2,6,1,5
	if (n_rec1 < 0) d3B2p = 0.;// slope_lim(pb[n][index_3D(n, i, j + D2, z - D3)][2], pb[n][index_3D(n, i, j + D2, z)][2], pb[n][index_3D(n, i, j + D2, z + D3)][2]);
	if (n_rec3 < 0) d3B2m = 0.;// slope_lim(pb[n][index_3D(n, i, j, z - D3)][2], pb[n][index_3D(n, i, j, z)][2], pb[n][index_3D(n, i, j, z + D3)][2]);

	d23B1p = 4.*(b1_5 + b1_8 - b1_6 - b1_7); //5,6,7,8
	d23B1m = 4.*(b1_1 + b1_4 - b1_2 - b1_3); //1,2,3,4
	d13B2p = 4.*(b2_3 + b2_8 - b2_4 - b2_7); //3,4,7,8
	d13B2m = 4.*(b2_1 + b2_6 - b2_2 - b2_5); //1,2,5,6
	d12B3p = 4.*(b3_2 + b3_8 - b3_4 - b3_6); //2,4,6,8
	d12B3m = 4.*(b3_1 + b3_7 - b3_3 - b3_5); //1,3,5,7

	B3p = pb[n][index_3D(n, i, j, z + D3)][3];
	B3m = pb[n][index_3D(n, i, j, z)][3];

	a123 = d23B1p - d23B1m;
	b123 = d13B2p - d13B2m;
	c123 = d12B3p - d12B3m;
	c233 = -a123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][2] * dx[n][3] * dx[n][3];;
	c133 = -b123 / 4. / (dx[n][1] * dx[n][2] * dx[n][3])*dx[n][1] * dx[n][3] * dx[n][3];;
	c1 = 0.5*(d1B3p + d1B3m) + b123 * dx[n][3] / (16.*dx[n][2]);
	c2 = 0.5*(d2B3p + d2B3m) + a123 * dx[n][3] / (16.*dx[n][1]);
	c3 = B3p - B3m;
	c12 = 0.5*(d12B3p + d12B3m);
	c13 = d1B3p - d1B3m;
	c23 = d2B3p - d2B3m;
	a13 = d3B1p - d3B1m;
	b23 = d3B2p - d3B2m;
	c33 = -0.5*(a13 / dx[n][1] + b23 / dx[n][2])*dx[n][3];
	c0 = 0.5*(B3p + B3m) - c33 / 4.;

	double var = c0 + c1*offset_1 + c2*offset_2 + c3*offset_3 + c13*offset_1*offset_3 + c23*offset_2*offset_3 + c33*offset_3*offset_3 +
		c12*offset_1*offset_2 + c233*offset_2*offset_3*offset_3 + c133*offset_1*offset_3*offset_3 + c123*offset_1*offset_2*offset_3;
	return var;
}