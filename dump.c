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

void dump(FILE *fp)
{
	int i,j,z,k ;
	int di, dj, dz;
	double i_double, j_double, z_double;
	double divb ;
	double X[NDIM] ;
	double r,th,phi,vmin,vmax ;
	struct of_geom geom ;
	struct of_state q ;
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	char newline[] = { '\n'};
	/***************************************************************
	  Write header information : 
	***************************************************************/
	if (rank == 0){
		int N1_print = N1;
		int N2_print = N2;
		int N3_print = N3;
		fwrite(&t, double_size, 1, fp);
		fwrite(&N1_print, int_size, 1, fp);
		fwrite(&N2_print, int_size, 1, fp);
		fwrite(&N3_print, int_size, 1, fp);
		fwrite(&n_rows, int_size, 1, fp);
		fwrite(&n_columns, int_size, 1, fp);
		fwrite(&n_stacks, int_size, 1, fp);
		fwrite(&startx[1], double_size, 1, fp);
		fwrite(&startx[2], double_size, 1, fp);
		fwrite(&startx[3], double_size, 1, fp);
		fwrite(&dx[1], double_size, 1, fp);
		fwrite(&dx[2], double_size, 1, fp);
		fwrite(&dx[3], double_size, 1, fp);
		fwrite(&tf, double_size, 1, fp);
		fwrite(&nstep, int_size, 1, fp);
		fwrite(&a, double_size, 1, fp);
		fwrite(&gam, double_size, 1, fp);
		fwrite(&cour, double_size, 1, fp);
		fwrite(&DTd, double_size, 1, fp);
		fwrite(&DTl, double_size, 1, fp);
		fwrite(&DTi, double_size, 1, fp);
		fwrite(&DTr, int_size, 1, fp);
		fwrite(&dump_cnt, int_size, 1, fp);
		fwrite(&image_cnt, int_size, 1, fp);
		fwrite(&rdump_cnt, int_size, 1, fp);
		fwrite(&dt, double_size, 1, fp);
		fwrite(&lim, int_size, 1, fp);
		fwrite(&failed, int_size, 1, fp);
		fwrite(&Rin, double_size, 1, fp);
		fwrite(&Rout, double_size, 1, fp);
		fwrite(&hslope, double_size, 1, fp);
		fwrite(&R0, double_size, 1, fp);
	}
	/***************************************************************
	  Write header information : 
	***************************************************************/
	di = (N1>1);
	dj = (N2>1);
	dz = (N3>1);
	ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) {
		coord(i,j,z,CENT,X) ;
		bl_coord(X,&r,&th,&phi) ;
		i_double=(double)i;
		j_double = (double)j;
		z_double = (double)z;
		fwrite(&i_double, double_size, 1, fp);
		fwrite(&j_double, double_size, 1, fp);
		fwrite(&z_double, double_size, 1, fp);
		fwrite(&X[1], double_size, 1, fp);
		fwrite(&X[2], double_size, 1, fp);
		fwrite(&X[3], double_size, 1, fp);
		fwrite(&r, double_size, 1, fp);
		fwrite(&th, double_size, 1, fp);
		fwrite(&phi, double_size, 1, fp);
		PLOOP fwrite(&(p[index(i, j, z)][k]), double_size, 1, fp);
		
        /* divb flux-ct defn; corner-centered.  Useonly interior corners */
		if(i > 0 && j > 0 && i < N1 && j < N2) {
			divb = fabs(
				#if(N1>1)
				0.25*(
				+p[index(i, j, z)][B1] * gdet[index2(i, j)][CENT]
				+ p[index(i, j, z - dz)][B1] * gdet[index2(i, j)][CENT]
				+ p[index(i, j - dj, z)][B1] * gdet[index2(i, j - dj)][CENT]
				+ p[index(i, j - dj, z - dz)][B1] * gdet[index2(i, j - dj)][CENT]
				- p[index(i - 1, j, z)][B1] * gdet[index2(i - 1, j)][CENT]
				- p[index(i - 1, j, z - dz)][B1] * gdet[index2(i - 1, j)][CENT]
				- p[index(i - 1, j - dj, z)][B1] * gdet[index2(i - 1, j - dj)][CENT]
				- p[index(i - 1, j - dj, z - dz)][B1] * gdet[index2(i - 1, j - dj)][CENT]
				) / dx[1]
				#endif
				#if(N2>1)
				+ 0.25*(
				+p[index(i, j, z)][B2] * gdet[index2(i, j)][CENT]
				+ p[index(i, j, z - dz)][B2] * gdet[index2(i, j)][CENT]
				+ p[index(i - di, j, z)][B2] * gdet[index2(i - di, j)][CENT]
				+ p[index(i - di, j, z - dz)][B2] * gdet[index2(i - di, j)][CENT]
				- p[index(i, j - 1, z)][B2] * gdet[index2(i, j - 1)][CENT]
				- p[index(i, j - 1, z - dz)][B2] * gdet[index2(i, j - 1)][CENT]
				- p[index(i - di, j - 1, z)][B2] * gdet[index2(i - di, j - 1)][CENT]
				- p[index(i - di, j - 1, z - dz)][B2] * gdet[index2(i - di, j - 1)][CENT]
				) / dx[2]
				#endif
				#if(N3>1)
				+ 0.25*(
				+p[index(i, j, z)][B3] * gdet[index2(i, j)][CENT]
				+ p[index(i - di, j, z)][B3] * gdet[index2(i - di, j)][CENT]
				+ p[index(i, j - dj, z)][B3] * gdet[index2(i, j - dj)][CENT]
				+ p[index(i - di, j - dj, z)][B3] * gdet[index2(i - di, j - dj)][CENT]
				- p[index(i, j, z - 1)][B3] * gdet[index2(i, j)][CENT]
				- p[index(i - di, j, z - 1)][B3] * gdet[index2(i - di, j)][CENT]
				- p[index(i, j - dj, z - 1)][B3] * gdet[index2(i, j - dj)][CENT]
				- p[index(i - di, j - dj, z - 1)][B3] * gdet[index2(i - di, j - dj)][CENT]
				) / dx[3]
				#endif
				);
		}
		else divb = 0. ;

		fwrite(&divb, double_size, 1, fp);

		if(!failed) {
			get_geometry(i,j,CENT,&geom) ;
			get_state(p[index(i,j,z)],&geom,&q) ;

			for (k = 0; k<NDIM; k++) fwrite(&(q.ucon[k]), double_size, 1, fp);
			for (k = 0; k<NDIM; k++) fwrite(&(q.ucov[k]), double_size, 1, fp);
			for (k = 0; k<NDIM; k++) fwrite(&(q.bcon[k]), double_size, 1, fp);
			for (k = 0; k<NDIM; k++) fwrite(&(q.bcov[k]), double_size, 1, fp);

			vchar(p[index(i,j,z)],&q,&geom,1,&vmax,&vmin) ;
			fwrite(&vmin, double_size, 1, fp);
			fwrite(&vmax, double_size, 1, fp);

			vchar(p[index(i,j,z)],&q,&geom,2,&vmax,&vmin) ;
			fwrite(&vmin, double_size, 1, fp);
			fwrite(&vmax, double_size, 1, fp);

			fwrite(&(geom.g), double_size, 1, fp);
		}
	}
}

void gdump(FILE *fp)
{
  int i,j,k,l ;
	double divb ;
	double X[NDIM] ;
	double r,th,phi,vmin,vmax ;
	double i_double, j_double, z_double;
	struct of_geom geom ;
	struct of_state q ;
	double tfac, rfac, hfac1, hfac2, pfac;
	double A1, A2, A3, Xc, sign;
	sign = 1.;
	double zero = 0.0;
	int z = 0; //The grid is axysymmetric, so put out only for z=0
	int int_size = sizeof(int);
	int double_size = sizeof(double);
	/***************************************************************
	  Write header information : 
	***************************************************************/
	if (rank == 0){
		int N1_print = N1;
		int N2_print = N2;
		int N3_print = N3;
		fwrite(&t, double_size, 1, fp);
		fwrite(&N1_print, int_size, 1, fp);
		fwrite(&N2_print, int_size, 1, fp);
		fwrite(&N3_print, int_size, 1, fp);
		fwrite(&n_rows, int_size, 1, fp);
		fwrite(&n_columns, int_size, 1, fp);
		fwrite(&n_stacks, int_size, 1, fp);
		fwrite(&startx[1], double_size, 1, fp);
		fwrite(&startx[2], double_size, 1, fp);
		fwrite(&startx[3], double_size, 1, fp);
		fwrite(&dx[1], double_size, 1, fp);
		fwrite(&dx[2], double_size, 1, fp);
		fwrite(&dx[3], double_size, 1, fp);
		fwrite(&tf, double_size, 1, fp);
		fwrite(&nstep, int_size, 1, fp);
		fwrite(&a, double_size, 1, fp);
		fwrite(&gam, double_size, 1, fp);
		fwrite(&cour, double_size, 1, fp);
		fwrite(&DTd, double_size, 1, fp);
		fwrite(&DTl, double_size, 1, fp);
		fwrite(&DTi, double_size, 1, fp);
		fwrite(&DTr, int_size, 1, fp);
		fwrite(&dump_cnt, int_size, 1, fp);
		fwrite(&image_cnt, int_size, 1, fp);
		fwrite(&rdump_cnt, int_size, 1, fp);
		fwrite(&dt, double_size, 1, fp);
		fwrite(&lim, int_size, 1, fp);
		fwrite(&failed, int_size, 1, fp);
		fwrite(&Rin, double_size, 1, fp);
		fwrite(&Rout, double_size, 1, fp);
		fwrite(&hslope, double_size, 1, fp);
		fwrite(&R0, double_size, 1, fp);
	}
	/***************************************************************
	  Write header information : 
	***************************************************************/
	ZSLOOP(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1)
	{
		coord(i,j,z,CENT,X) ;
		bl_coord(X,&r,&th, &phi) ;

		//drdx
		tfac = 1. ;
		rfac = r - R0 ;
		A1 = CHARLIE*DELTA*pow(log(10.), -DELTA)*pow(CHARLIE*log(r), -1. + DELTA) / pow(1. + pow(log(10.), -DELTA)*pow(CHARLIE*log(r), DELTA), 2.);
		A2 = (0.5 - BRAVO*0.5) / pow(0.5, QUEBEC);
		A3 = 1. / (1. + pow(log(10.), -DELTA)*pow(CHARLIE*log(r), DELTA));
		Xc = sqrt(pow(X[2], 2.));
		sign = 1.;
		if (X[2] < 0.0){
			sign = -1.;
		}
		if (X[2] > 1.0){
			sign = -1.;
			Xc = 2. - Xc;
		}
		if (X[2] < 0.5){
			hfac1 = sign*(-M_PI*A1*Xc + M_PI*A1*(BRAVO*Xc + A2*pow(Xc, QUEBEC)) + 0.5*A1*sin(2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
			hfac2 = M_PI * A3 + M_PI*(BRAVO + A2*QUEBEC*pow(Xc, -1. + QUEBEC))*(1. - A3) *(1. - cos(2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
		}
		else{
			hfac1 = sign*(-M_PI*A1*(1. - Xc) + M_PI*A1*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC)) + 0.5*A1*sin(2.*M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))));
			hfac2 = M_PI * A3 + M_PI*(BRAVO + A2*QUEBEC*pow(1. - Xc, -1. + QUEBEC))*(1. - A3) *(1. - cos(2.*M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))));
		}
		pfac = 1. ;

		i_double = (double)i;
		j_double = (double)j;
		z_double = (double)z;
		fwrite(&i_double, double_size, 1, fp);
		fwrite(&j_double, double_size, 1, fp);
		fwrite(&z_double, double_size, 1, fp);
		fwrite(&X[1], double_size, 1, fp);
		fwrite(&X[2], double_size, 1, fp);
		fwrite(&X[3], double_size, 1, fp);
		fwrite(&r, double_size, 1, fp);
		fwrite(&th, double_size, 1, fp);
		fwrite(&phi, double_size, 1, fp);
		if(!failed) {
			get_geometry(i,j,CENT,&geom) ;

			//g_{kl}
			for(k=0;k<NDIM;k++) 
			  for(l=0;l<NDIM;l++) 
				  fwrite(&(geom.gcov[k][l]), double_size, 1, fp);

			//g^{kl}
			for(k=0;k<NDIM;k++) 
			  for(l=0;l<NDIM;l++) 
				  fwrite(&(geom.gcon[k][l]), double_size, 1, fp);

			//(-deg(g))**0.5
			fwrite(&(geom.g), double_size, 1, fp);

			//dr^i/dx^j
			for(k=0;k<NDIM;k++) {
			  for(l=0;l<NDIM;l++) {
			    if(k==0 && l==0){
					fwrite(&tfac, double_size, 1, fp);
			    }
			    else if(k==1 && l==1){
					fwrite(&rfac, double_size, 1, fp);
			    }
				else if (k==2 && l==1){
					fwrite(&hfac1, double_size, 1, fp);
				}
				else if (k==1 && l==2){
					fwrite(&hfac1, double_size, 1, fp);
				}
			    else if(k==2 && l==2){
					fwrite(&hfac2, double_size, 1, fp);
			    }
			    else if(k==3 && l==3){
					fwrite(&pfac, double_size, 1, fp);
			    }
			    else{
					fwrite(&zero, double_size, 1, fp);
			    }
			  }
			}
		}
	}
}
