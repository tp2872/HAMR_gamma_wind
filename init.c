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
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURpos_newE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/

/*
 *
 * generates initial conditions for a fishbone & moncrief disk 
 * with exterior at minimum values for density & internal energy.
 *
 * cfg 8-10-01
 *
 */
#include <float.h>
#include "decs_MPI.h"

void rotate_vector2(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt);
void coord_transform(double *pr, int n, int ii, int jj, int zz);
void set_mag(void);
void rotate_vector2(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt);
void coord_transform(double *pr, int n, int ii, int jj, int zz);
void set_mag(void);
void init_thindisk();
double compute_Amax(double(*restrict A[NB])[NPR]);
double compute_B_from_A(void);
double normalize_B_by_maxima_ratio(double beta_target, double *norm_value);
double normalize_B_by_beta(double beta_target, double rmax, double *norm_value);
double rtbis(double(*func)(double, double*), double *parms, double x1, double x2, double xacc);
double lfunc(double lin, double *parms);
void compute_gu(double r, double th, double a, double *gutt, double *gutp, double *gupp);
double thintorus_findl(double r, double th, double a, double c, double al);
double compute_udt(double r, double th, double a, double l);
double compute_omega(double r, double th, double a, double l);
double compute_uuphi(double r, double th, double a, double l);
double compute_l_from_omega(double r, double th, double a, double omega1);
void getmax_densities(double(*restrict prim[NB])[NPR], double *rhomax, double *umax);
double get_maxprimvalrpow(double(*restrict prim[NB])[NPR], double rpow, int m);
int normalize_field_local_nodivb(double targbeta, double rhomax, double amax, double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR], int dir);
double compute_rat(double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR], double rhomax, double amax, double targbeta, int loc, int n, int i, int j, int k);
double compute_profile(double(*restrict prim[NB])[NPR], double amax, double aphipow, int loc, int n, int i, int j, int z);
int compute_vpot_from_gdetB1(double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR]);
void get_rho_u_floor(double r, double th, double phi, double *rho_floor, double *u_floor);
void init_torus_grb();
void set_mag_TDE(void);
void set_uniform_Bphi(void);
double lfish_calc(double r);

double global_kappa, aphipow;

#ifndef M_PI_2
#define M_PI_2 (M_PI/2.)
#endif
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131E-16
#endif
#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

typedef struct {
  double xmin, xmax, ymin, ymax, zmin, zmax; //array extent
  int nvars, nx, ny, nz; //resolution
} extent;


void init()
{
	void init_bondi(void);
	void init_torus(void);
	void init_torus_grb();
	void init_disruption(void);
	void init_monopole(double Rout_val);
	void init_thindisk();

	switch( WHICHPROBLEM ) {
		case MONOPOLE_PROBLEM_1D:
		case MONOPOLE_PROBLEM_2D:
		init_monopole(1e3);
		break;
		case BZ_MONOPOLE_2D:
		init_monopole(100.);
		break;
		case TORUS_PROBLEM:
		init_torus();
		break;
		case THIN_PROBLEM:
		init_thindisk();
		break;
		case BONDI_PROBLEM_1D:
		init_thindisk();
		break;
		case DISRUPTION_PROBLEM:
		init_disruption();
		break;
		case TORUS_PROBLEM_GRB:
		init_torus_grb();
		break;
		case BONDI_PROBLEM_2D:
		init_bondi();
		break;
	}

	int n;
	#if(GPU_ENABLED || GPU_DEBUG )
	GPU_boundprim(1);
	#endif
}

void init_thindisk()
{
	int i, j, z, n;
	double r, th, phi, sth, cth;
	double ur, uh, up, u, rho;
	double bl_gcov[NDIM][NDIM];
	double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
	double tilt, eccentricity;
	struct of_geom geom;

	/* for disk interior */
	double l, rin, lnh, expm2chi, up1;
	double DD, AA, SS, thin, sthin, cthin, DDin, AAin, SSin;
	double kappa, hm1;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av, rhomax, umax, beta, bsq_ij, bsq_max, norm, q, beta_act;

	/* disk parameters (use fishbone.m to select new solutions) */
	double temp = a;
	a = 0.9375;
	rin = 6.5;
	rmax = 80.;
	kappa = 1.e-3;
	beta = 100.;

	coord(0, 5, 0, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);
	if (rank == 0) {
		fprintf(stderr, "r[5]: %g\n", r);
		fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - a * a)));
		if (r > 1. + sqrt(1. - a * a)) {
			fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n");
		}
		else {
			fprintf(stderr, "\n");
		}
	}

	/* output choices */
	tf = 200000000.0;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;

	rhomax = 0.;
	umax = 0.;
	#if(!NSY)
	tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif
	eccentricity = 0.0;
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa,hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;
			#if (TILTED)
			sph_to_cart(X_cart, &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
			rotate_coord(X_cart, -tilt);
			cart_to_sph(X_cart, &r, &th, &phi);
			#endif

			#if(ELLIPTICAL)
			sph_to_cart(X_cart, &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
			elliptical_coord(X_cart, pos_new, &r, eccentricity);
			#endif

			sth = sin(th);
			cth = cos(th);

			double rhoc;
			thin = M_PI / 2.;
			double A, R, D, E, L;
			A = 1.0 + a*a / (r*r) + 2.0*a*a / (r*r*r);
			R = 1 + a / (r*sqrt(r));
			D = 1.0 - 2.0 / r + a*a / (r*sqrt(r));
			E = 1.0 + 4.0*a*a / (r*r) - 4.0*a*a / (r*r*r) + 3.0 * a*a*a*a / (r*r*r*r);
			L = 1.;
			rhoc = 1./r*pow(A, -4.)*pow(R, 6.0)*D*E*E / (L*L);
			if (r > rmax) rhoc = 0.0;////rhoc /= exp(sqrt(r-rmax));
			rho = rhoc * exp(-pow((th-thin)/H_OVER_R,2.0)*0.5);
			ur = 0.;
			uh = 0.;
			up = 0.;
			/* regions outside torus */
			if (r > 4*rmax || r < 2.0) {
				rho = 1.e-7*RHOMIN;
				u = 1.e-7*UUMIN;



				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;
			}
			/* region inside magnetized torus; u^i is calculated in
			* Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			* so it needs to be transformed at the end */
			else {
				up = 1. / (pow(r, 3. / 2.) + a);
				up *= sqrt(1. / (1 - up*up));
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;

				if (rho > rhomax) {
					#pragma omp critical
					rhomax = rho;
				}

				double T_target = M_PI / 2.*pow(H_OVER_R*r*up, 2.);
				u = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * T_target / (gam - 1.);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u * (1. + 4.e-2*(ranc(0) - 0.5));
				if (u > umax && r > rin) {
					#pragma omp critical
					umax = u;
				}

				#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
				rotate_vector(V, pos_new, &r, &th, &phi, tilt);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#elif(ELLIPTICAL)
				V_old[1] = ur;
				V_old[2] = uh;
				V_old[3] = up;
				elliptical_vector(X_cart, V_old, V_new, pos_new, &r, &th, eccentricity);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V_new[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V_new[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V_new[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#else
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;//watch out

																	  /* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#endif
			}
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
		}
	}
	a = temp;
	#if (MPI_enable)
	/*Share rhomax among MPI processes*/
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);

	/*Share umax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0) {
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	//ZSLOOP(0,N1-1,0,N2-1) {
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] /= rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] /= rhomax;
		}
	}
	umax /= rhomax;
	rhomax = 1.;
	for (n = 0; n < n_active; n++) {
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);

	set_mag();

	#if( DO_FONT_FIX ) 
	set_Katm();
	#endif 

	sourceflag = 0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif
}

void init_torus()
{
	int i,j,z,n ;
	double r,th,phi,sth,cth ;
	double ur,uh,up,u,rho ;
	double bl_gcov[NDIM][NDIM];
	double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
	double tilt, eccentricity;
	struct of_geom geom ;

	/* for disk interior */
	double l,rin,lnh,expm2chi,up1 ;
	double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
	double kappa,hm1 ;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;

	/* disk parameters (use fishbone.m to select new solutions) */
	double temp = a;
	a = 0.9375;
	rin = 6.0;
	rmax = 12.;
    l = lfish_calc(rmax) ;
	kappa = 1.e-3 ;
	beta = 100. ;

	coord(0,5, 0, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);
	if (rank == 0) {
		fprintf(stderr, "r[5]: %g\n", r);
		fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - a*a)));
		if (r > 1. + sqrt(1. - a*a)) {
			fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n");
		}
		else {
			fprintf(stderr, "\n");
		}
	}
	
    /* output choices */
	tf = 200000000.0 ;

	/* start diagnostic counters */
	dump_cnt = 0 ;
	dump_cnt_reduced = 0;
	image_cnt = 0 ;
	rdump_cnt = 0 ;

	rhomax = 0. ;
	umax = 0. ;
	#if(!NSY)
	tilt = (TILT_ANGLE)/180.*M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif
	eccentricity = 0.0;
	for (n = 0; n < n_active; n++){
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa,hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X,&r,&th, &phi) ;
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;
			#if (TILTED)
			sph_to_cart(X_cart,  &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
			rotate_coord(X_cart,-tilt);
			cart_to_sph(X_cart, &r, &th, &phi);
			#endif

			#if(ELLIPTICAL)
			sph_to_cart(X_cart, &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
			elliptical_coord(X_cart,pos_new, &r, eccentricity);
			#endif

			sth = sin(th) ;
			cth = cos(th) ;

			/* calculate lnh */
			DD = r*r - 2.*r + a*a ;
			AA = (r*r + a*a)*(r*r + a*a) - DD*a*a*sth*sth ;
			SS = r*r + a*a*cth*cth ;

			thin = M_PI/2. ;
			sthin = sin(thin) ;
			cthin = cos(thin) ;
			DDin = rin*rin - 2.*rin + a*a ;
			AAin = (rin*rin + a*a)*(rin*rin + a*a) 
				- DDin*a*a*sthin*sthin ;
			SSin = rin*rin + a*a*cthin*cthin ;

			if(r >= rin) {
				lnh = 0.5*log((1. + sqrt(1. + 4.*(l*l*SS*SS)*DD/
					(AA*sth*AA*sth)))/(SS*DD/AA)) 
					- 0.5*sqrt(1. + 4.*(l*l*SS*SS)*DD/(AA*AA*sth*sth))
					- 2.*a*r*l/AA 
					- (0.5*log((1. + sqrt(1. + 4.*(l*l*SSin*SSin)*DDin/
					(AAin*AAin*sthin*sthin)))/(SSin*DDin/AAin)) 
					- 0.5*sqrt(1. + 4.*(l*l*SSin*SSin)*DDin/
						(AAin*AAin*sthin*sthin)) 
					- 2.*a*rin*l/AAin ) ;
			}
			else
				lnh = 1. ;

			/* regions outside torus */
			if(lnh < 0. || r < rin) {
				rho = 1.e-7*RHOMIN ;
				u = 1.e-7*UUMIN ;

				ur = 0. ;
				uh = 0. ;
				up = 0. ;

				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][RHO] = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][UU] = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U3] = up;
			}
			/* region inside magnetized torus; u^i is calculated in
			 * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			 * so it needs to be transformed at the end */
			else { 
				hm1 = exp(lnh) - 1. ;
				rho = pow(hm1*(gam - 1.)/(kappa*gam),
							1./(gam - 1.)) ; 
				u = kappa*pow(rho,gam)/(gam - 1.) ;
				ur = 0. ;
				uh = 0. ;

				/* calculate u^phi */
				expm2chi = SS*SS*DD/(AA*AA*sth*sth) ;
				up1 = sqrt((-1. + sqrt(1. + 4.*l*l*expm2chi))/2.) ;
				up = 2.*a*r*sqrt(1. + up1*up1)/sqrt(AA*SS*DD) +
					sqrt(SS/AA)*up1/sth ;

				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][RHO] = rho;

				if (rho > rhomax){
					#pragma omp critical
					rhomax = rho;
				}
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][UU] = u*(1. + 4.e-2*(ranc(0) - 0.5));
				if(u > umax && r > rin){
					#pragma omp critical
					umax = u ;
				}
			
				#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
				//th = (th - M_PI / 2.) *(fractheta*0.5) + M_PI / 2.;
				rotate_vector(V, pos_new, &r, &th, &phi, tilt);
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U1] = V[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U2] = V[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U3] = V[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)], n_ord[n], i, j, z);
				#elif(ELLIPTICAL)
				V_old[1] = ur;
				V_old[2] = uh;
				V_old[3] = up;
				elliptical_vector(X_cart, V_old, V_new, pos_new, &r, &th, eccentricity);
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U1] = V_new[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U2] = V_new[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U3] = V_new[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)], n_ord[n], i, j, z);
				#else
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][U3] = up;//watch out

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], n_ord[n], i, j, z);
				#endif
			}
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B3] = 0.;
		}
	}
	a = temp;
	#if (MPI_enable)
	/*Share rhomax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);

	/*Share umax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#endif

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0){
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	//ZSLOOP(0,N1-1,0,N2-1) {
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][RHO] /= rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][UU] /= rhomax;
		}
	}
	umax /= rhomax ;
	rhomax = 1. ;
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}

	bound_prim(p, 1);

	set_mag();

	#if( DO_FONT_FIX ) 
	set_Katm();
	#endif 

	sourceflag=0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif
}

void init_disruption()
{
  int interpolate_prims( double r, double th, double ph, extent ext, double *data, double *p);
  int i,j,z,n ;
  double r,th,phi,sth,cth ;
  double ur,uh,up,u,rho ;
  double bl_gcov[NDIM][NDIM];
  double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
  double tilt, eccentricity;
  struct of_geom geom ;
  
  /* for disk interior */
  double l,rin,lnh,expm2chi,up1 ;
  double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
  double kappa,hm1 ;
  
  /*For MPI*/
  double inmsg;
  
  /* for magnetic field */
  double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;
  
  /* for ICs */
  FILE *fp;
  int ind;
# define MAXLEN (1024)
  extent ext;
  int res;
  double *icdata;
  char fname[] = "icdata", headerstr[MAXLEN];
  size_t memsize, nitems, nread;
  double prim[NPR];
  int k;
  
  /* disk parameters (use fishbone.m to select new solutions) */
  a = 0.9375 ;
  
  coord(0,5, 0, 0, CENT, X);
  bl_coord(X, &r, &th, &phi);
  if (rank == 0) {
    fprintf(stderr, "r[5]: %g\n", r);
    fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - a*a)));
    if (r > 1. + sqrt(1. - a*a)) {
      fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n");
    }
    else {
      fprintf(stderr, "\n");
    }
  }
  
  /* output choices */
  tf = 200000000.0 ;
  
  /* start diagnostic counters */
  dump_cnt = 0 ;
  dump_cnt_reduced = 0;
  image_cnt = 0 ;
  rdump_cnt = 0 ;

  //read ICs from file
  //for this, loop over all MPI processes
  //and let them read the IC data from file, one by one
  for (ind=0; ind<numtasks; ind++) {
    if (ind == rank) {
      fp = fopen(fname, "rb");
	  if (NULL == fp && 0 == rank) {
        fprintf(stderr, "Could not open file %s for reading, exiting\n", fname);
        exit(1234);
      }
      fgets(headerstr, MAXLEN, fp);
      sscanf(headerstr, "#%d %d %d %d %lf %lf %lf %lf %lf %lf ",
             &ext.nvars, &ext.nx, &ext.ny, &ext.nz,
             &ext.xmin, &ext.xmax, &ext.ymin, &ext.ymax, &ext.zmin, &ext.zmax);
      if (0 == rank) {
        fprintf(stderr, "[%d] reading IC block: resolution (%dx%dx%dx%d), extent (%g,%g)x(%g,%g)x(%g,%g), file %s...",
                rank,
                ext.nvars, ext.nx, ext.ny, ext.nz,
                ext.xmin, ext.xmax,
                ext.ymin, ext.ymax,
                ext.zmin, ext.zmax,
                fname);
        fflush(stderr);
      }
      nitems = (size_t)ext.nvars*ext.nx*ext.ny*ext.nz;
      memsize = sizeof(double)*nitems;
      icdata = malloc(memsize);
      if(NULL == icdata) {
        fprintf(stderr,"[%5d] could not allocate memory of size %ld\n", rank, memsize);
        fclose(fp);
        exit(1235);
      }
      //read in the data block from file
      nread = fread(icdata, sizeof(double), nitems, fp);
      fclose(fp);
      fp = NULL;
      if (nread != nitems) {
        fprintf( stderr, "[%d] error reading from %s: items expected %ld, written %ld\n", rank, fname, nitems, nread);
        exit(1236);
      }
      if (0 == rank) {
        fprintf(stderr, " done\n");
        fflush(stderr);
      }
      //now icdata contains the IC information
    }

  }

  //vars: [x],[y],[z],[rho],[ug],[vx],[vy],[vz],[poten]
  //ivar:  0,  1,  2,   3,   4,   5,   6,   7,     8
  //mapping: icdata[((ivar*nx+ii)*ny+jj)*nz+kk] 
  rhomax = 0. ;
  umax = 0. ;
	#if(!NSY)
  tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
  tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif  
  eccentricity = 0.0;
  for (n = 0; n < n_active; n++){
    ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
      coord(n_ord[n], i, j, z, CENT, X);
      bl_coord(X,&r,&th, &phi) ;
      pos_new[1] = r;
      pos_new[2] = th;
      pos_new[3] = phi;
      
      sth = sin(th) ;
      cth = cos(th) ;
      
      res = interpolate_prims(r, th, phi, ext, icdata, prim);

	  /* regions outside stream */
      if(res ||prim[RHO] < 1e-20 || r<10) {
        rho = 1.e-20;
        u = 1.e-20;
        
        ur = 0. ;
        uh = 0. ;
        up = 0. ;
        
		prim[RHO] = rho;
		prim[UU] = u;
        prim[U1] = ur;
        prim[U2] = uh;
        prim[U3] = up;
      }
      else {
        /* convert from BL 4-vel to relative 4-vel in internal (KS prime) coords */
        coord_transform(prim, n_ord[n], i, j, z);
      }
	  //if (prim[RHO] < 0.01) prim[RHO] = 0.0;
      prim[B1] = 0.;
      prim[B2] = 0.;
      prim[B3] = 0.;
      //copy back to full prim array
      PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k] = prim[k];
      if(prim[RHO]>rhomax) {
        rhomax = prim[RHO];
      }
    }
  }
  if(icdata) {
    free(icdata);
    icdata = NULL;
  }
  #if (MPI_enable)
  /*Share rhomax among MPI processes*/
  MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
  
  /*Share umax among MPI processes*/
  MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
  #endif
  
  /* Normalize the densities so that max(rho) = 1 */
  if (rank == 0){
    fprintf(stderr, "rhomax: %g\n", rhomax);
  }
  //ZSLOOP(0,N1-1,0,N2-1) {
  for (n = 0; n < n_active; n++){
    ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
      p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][RHO] /= rhomax;
      p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][UU] /= rhomax;
    }
  }
  umax /= rhomax ;
  rhomax = 1. ;
  for (n = 0; n < n_active; n++){
    fixup(p, n_ord[n]);
  }
  bound_prim(p,1);

  //set_mag();
  
	#if( DO_FONT_FIX ) 
  set_Katm();
	#endif 
  
  sourceflag=0.;
	#if(ELLIPTICAL2)
  calc_source();
	#endif
}

int interpolate_prims( double r, double th, double ph, extent ext, double *data, double *p)
{
  int interpolate_var( double r, double th, double ph, extent ext, double *data, int ivar, double *val);
  double vx, vy, vz, poten, x, y, z, R;
  double bl_gcov[NDIM][NDIM];
  int res;
  //vars: [x],[y],[z],[rho],[ug],[vx],[vy],[vz],[poten]
  //ivar:  0,  1,  2,   3,   4,   5,   6,   7,     8
  res = interpolate_var(r,th,ph,ext,data,3,&p[RHO]);
  if(res) return(res);
  res = interpolate_var(r,th,ph,ext,data,4,&p[UU]);
  res = interpolate_var(r,th,ph,ext,data,5,&vx);
  res = interpolate_var(r,th,ph,ext,data,6,&vy);
  res = interpolate_var(r,th,ph,ext,data,7,&vz);
  res = interpolate_var(r,th,ph,ext,data,8,&poten);
  double x1, y1, z1;
  x = r*sin(th)*cos(ph);
  y = r*sin(th)*sin(ph);
  z = r*cos(th);
  R = sqrt(x*x+y*y);
  //Matthew: seems wrong to me, we work in a coordinate basis with boyer lindquist coordinates(r, theta,phi)!!!
 // bl_gcov_func(r, th, bl_gcov);

  if (vx*vx + vy*vy + vz*vz>1.0){
	  poten = vx*vx + vy*vy + vz*vz;
	  vx /= poten;
	  vy /= poten;
	  vz /= poten;
  }
  p[U1] = (vx*x + vy*y + vz*z) / r;             //dr/dt = dr/dx*vx + dr/dy*vy + dr/dz*vz
  p[U2] = (x*z*vx + y*z*vy - R*R*vz) / (r*r*R); //dth/dt = dth/dx*vx + dth/dy*vy + dth/dz*vz
  p[U3] = (-y*vx + x*vy) / (R*R);             //dph/dt = dph/dx*vx + dph/dy*vy + dph/dz*vz
  p[UU] = p[UU]*p[RHO];

//  p[U1] = (vx * sin(th)*cos(ph) + vy * sin(th)*sin(ph) + vz * cos(th)) / sqrt(bl_gcov[1][1]);
 // p[U2] = (vx * cos(th)*cos(ph) + vy * cos(th)*sin(ph) - vz * sin(th)) / sqrt(bl_gcov[2][2]);
 /// p[U3] = (-vx * sin(ph) + vy * cos(ph)) / sqrt(bl_gcov[3][3]);

 
  p[B1] = 0.;
  p[B2] = 0.;
  p[B3] = 0.;

  return(0);
}

//define compact form for array indexing
#define d(ii,jj,kk) data[((ivar*nx+ii)*ny+jj)*nz+kk]

int interpolate_var( double r, double th, double ph, extent ext, double *data, int ivar, double *val)
{
  double x, y, z, dx, dy, dz;
  double i, j, k, di, dj, dk;
  int i0, j0, k0, i1, j1, k1, nx, ny, nz;
  double c00, c01, c10, c11, c0, c1, c;

  nx = ext.nx;
  ny = ext.ny;
  nz = ext.nz;
  x = r*sin(th)*cos(ph);
  y = r*sin(th)*sin(ph);
  z = r*cos(th);
  dx = (ext.xmax-ext.xmin)/(nx-1);
  dy = (ext.ymax-ext.ymin)/(ny-1);
  dz = (ext.zmax-ext.zmin)/(nz-1);
  i = (x-ext.xmin)/dx;
  j = (y-ext.ymin)/dy;
  k = (z-ext.zmin)/dz;
  i0 = floor(i);
  j0 = floor(j);
  k0 = floor(k);
  i1 = (int)ceil(i);
  j1 = (int)ceil(j);
  k1 = (int)ceil(k);
  if(i0<5 || i1>=nx-5 || j0<5 || j1>=ny-5 || k0<5 || k1>=nz-5) {
    return(1);
  }
  di = i - floor(i);
  dj = j - floor(j);
  dk = k - floor(k);
  c00 = d(i0,j0,k0)*(1-di) + d(i1,j0,k0)*di;
  c01 = d(i0,j0,k1)*(1-di) + d(i1,j0,k1)*di;
  c10 = d(i0,j1,k0)*(1-di) + d(i1,j1,k0)*di;
  c11 = d(i0,j1,k1)*(1-di) + d(i1,j1,k1)*di;
  c0 = c00*(1-dj) + c10*dj;
  c1 = c01*(1-dj) + c11*dj;
  c = c0*(1-dk) + c1*dk;
  if (isnan(c))  return(1);
  *val = c;
  return(0);
  
}
//undefine array shortcut to avoid name conflicts
#undef d

void set_mag(void){
	int i, j, z, k, n;
	double rhomax = 1., umax = 0.;
	int i100 = 0;
	double rho_av, q, beta = 10.0, bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM],pos_new[NDIM], beta_ij;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	#if(!NSY)
	double tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
	double tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif	

	do{
		i100++;
		coord(0, i100, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
	} while (r < 400.0);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]]-N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3+D3){
			dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][0] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
		}
	}

	/* first find corner-centered vector potential */
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]]-N1G, BS_1 + N1_GPU_offset[n_ord[n]]+D1, N2_GPU_offset[n_ord[n]]-N2G, N2_GPU_offset[n_ord[n]] + BS_2+D2, N3_GPU_offset[n_ord[n]]-N3G, N3_GPU_offset[n_ord[n]] + BS_3+D3){
			/* Cell centered vector potential */	
			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax-0.0005;
			#else
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.05;
			#endif
			if (q > 0.){		
				coord(n_ord[n], i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				#if(WHICHPROBLEM==THIN_PROBLEM)
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = q*pow(r,2.0); //Toroidal
				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = sin(2.0*M_PI *r/120.)*sqrt(r*r*r*r*r)*q;
				#else
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = pow(q, 2.0) * pow(r, 3.0); //MAD
				#endif
			}
			else{
				dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] = 0.0;
			}
			if (q > 0.) {
				#if (TILTED)
				coord(n_ord[n],i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				pos_new[1] = r;
				pos_new[2] = th;
				pos_new[3] = phi;
				sph_to_cart(X_cart, &r, &th, &phi);
				rotate_coord(X_cart, -tilt);
				cart_to_sph(X_cart, &r, &th, &phi);

				V[1] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1];
				V[2] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2];
				V[3] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3];
				rotate_vector2(V, pos_new, &r, &th, &phi, tilt);
				dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] = V[1];
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = V[2];
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = V[3];
				#endif
			}
		}
	}

	//Transform from cell centered vector potential to edge centered vector potential
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - D1, BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] - D2, N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]] - D3, N3_GPU_offset[n_ord[n]] + BS_3){
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.25*(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z - D3)][1]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.25*(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z - D3)][2]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.25*(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j - D2, z)][3]);
		}
	}

	/* now differentiate to find cell-centered B,
	and begin normalization */
	#if(STAGGERED)
	gpu = 0;
	nstep = AMR_SWITCHTIMELEVEL - 1;
	set_prestep();
	const_transport_bound();
	nstep = 0;
	#endif
	for (n = 0; n < n_active; n++){
		#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = 0.;
			}
		}

		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.;
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = 0.;
			}
		}

		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
			get_geometry(n_ord[n], i, j, z, FACE1, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = -(E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][3]) / (dx[nl[n_ord[n]]][2] * geom.g)
				#if(N3G>0)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][2]) / (dx[nl[n_ord[n]]][3] * geom.g)
				#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE2, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][3]) / (dx[nl[n_ord[n]]][1] * geom.g)
				#if(N3G>0)
				- (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][1]) / (dx[nl[n_ord[n]]][3] * geom.g)
				#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE3, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = -(E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][2]) / (dx[nl[n_ord[n]]][1] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][1]) / (dx[nl[n_ord[n]]][2] * geom.g);
		}
			#endif
	}

	double bsq_max = 0.;
	double ug_sum = 0.;
	double bsq_sum = 0.;
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B1] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[nl[n_ord[n]]][2] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B2] =
				(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[nl[n_ord[n]]][1] * geom.g)
				- (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B3] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[nl[n_ord[n]]][1] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[nl[n_ord[n]]][2] * geom.g);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE1] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i + D1, j, z)][FACE1]) / (2.0* gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE2] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j + D2, z)][FACE2]) / (2.0* gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#if(N3G>0)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE3] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + D3)][FACE3]) / (2.0* gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			#endif
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#endif
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);
			beta_ij = 0.5*(gam - 1.0)*p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] / bsq_ij;
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > umax && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)){
				umax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			if (bsq_ij > bsq_max && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				bsq_max = bsq_ij;
			}
			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r<50.) {
				bsq_sum += bsq_ij* geom.g* dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

	#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	#endif
	#endif

	if (rank == 0){
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}

	/* finally, normalize to set field strength */
	#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.)*ug_sum / (0.5*bsq_sum);
	#else
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	#endif
	if (rank == 0){
		fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	}
	norm = sqrt(beta_act / beta);

	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= norm;
			#if(STAGGERED)
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= norm;
			#endif
		}
	}

	bsq_max = 0.;
	umax = 0;
	bsq_sum = 0.;
	ug_sum = 0.;
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);
			if (bsq_ij > bsq_max && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				bsq_max = bsq_ij;
			}
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > umax && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				umax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}

			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r<50.) {
				bsq_sum += bsq_ij* geom.g* dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

	/*Share bsq_max among MPI processes*/
	#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	#endif
	#endif

	#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.)*ug_sum / (0.5*bsq_sum);
	#else
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	#endif
	if (rank == 0){
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}

	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
}

void init_monopole(double Rout_val)
{
	fprintf(stderr, "Error. Monopole not implemented in this version\n");
}

double lfish_calc(double r)
{
	return(
		((pow(a, 2) - 2.*a*sqrt(r) + pow(r, 2))*
		((-2.*a*r*(pow(a, 2) - 2.*a*sqrt(r) + pow(r, 2))) /
		sqrt(2.*a*sqrt(r) + (-3. + r)*r) +
		((a + (-2. + r)*sqrt(r))*(pow(r, 3) + pow(a, 2)*(2. + r))) /
		sqrt(1 + (2.*a) / pow(r, 1.5) - 3. / r))) /
		(pow(r, 3)*sqrt(2.*a*sqrt(r) + (-3. + r)*r)*(pow(a, 2) + (-2. + r)*r))
		);
}

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
	blgset(n, ii, jj, &geom);

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
	double bl_gcov[NDIM][NDIM], gdet1, gdet2;
	double V_tmp[NDIM], X_tmp[NDIM], pos_new_tmp[NDIM];
	int i;
	for (i = 1; i < NDIM; i++){
		V_tmp[i] = V[i];
		pos_new_tmp[i] = pos_new[i];
	}

	bl_gcov_func(*r, *th, bl_gcov);

	V_tmp[1] *= sqrt(bl_gcov[1][1]);
	V_tmp[2] *= sqrt(bl_gcov[2][2]);
	V_tmp[3] *= sqrt(bl_gcov[3][3]);
	//V_tmp[3] = sqrt(bl_gcov[3][3] * V_tmp[3] * V_tmp[3]+2.*bl_gcov[0][3] * V_tmp[0] * V_tmp[3]);

	X_tmp[1] = V_tmp[1] * sin(*th)*cos(*phi) + V_tmp[2] * cos(*th)*cos(*phi) - V_tmp[3] * sin(*phi);
	X_tmp[2] = V_tmp[1] * sin(*th)*sin(*phi) + V_tmp[2] * cos(*th)*sin(*phi) + V_tmp[3] * cos(*phi);
	X_tmp[3] = V_tmp[1] * cos(*th) - V_tmp[2] * sin(*th);

	rotate_coord(X_tmp, tilt);

	bl_gcov_func(pos_new[1], pos_new[2], bl_gcov);
	//gdet2 = gdet_func(bl_gcov);
	V[0] = V_tmp[0];
	V[1] = (X_tmp[1] * sin(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * sin(pos_new[2])*sin(pos_new[3]) + X_tmp[3] * cos(pos_new[2])) / sqrt(bl_gcov[1][1]);
	V[2] = (X_tmp[1] * cos(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * cos(pos_new[2])*sin(pos_new[3]) - X_tmp[3] * sin(pos_new[2])) / sqrt(bl_gcov[2][2]);
	V[3] = (-X_tmp[1] * sin(pos_new[3]) + X_tmp[2] * cos(pos_new[3])) / sqrt(bl_gcov[3][3]);
	//V[3] = (-bl_gcov[0][3] * V[0] + V[3] / fabs(V[3])*sqrt(pow(bl_gcov[0][3] * V[0], 2.) + bl_gcov[3][3]*pow(V[3],2.))) / bl_gcov[3][3];
}


/*Calculates covariant vector components after vector is rotated from (r, th, phi) to (pos_new[1], pos_new[2], pos_new[3]) over angle tilt*/
void rotate_vector2(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt){
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
	X[1] = pow(log(*r - RB), 1. / RADEXP);
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
	invert_matrix(bl_gcov, bl_gcon);
	dxdxp_func(X, dxdxp);
	invert_matrix(dxdxp, dxpdx);

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
	V_tmp[1] *= sqrt(bl_gcon1[1][1]);
	V_tmp[2] *= sqrt(bl_gcon1[2][2]);
	V_tmp[3] *= sqrt(bl_gcon1[3][3]);

	/*Calculate Cartesian components (x, y, z) at pos_newition (r, th, phi) of vector V*/
	X_tmp[1] = V_tmp[1] * sin(*th)*cos(*phi) + V_tmp[2] * cos(*th)*cos(*phi) - V_tmp[3] * sin(*phi);
	X_tmp[2] = V_tmp[1] * sin(*th)*sin(*phi) + V_tmp[2] * cos(*th)*sin(*phi) + V_tmp[3] * cos(*phi);
	X_tmp[3] = V_tmp[1] * cos(*th) - V_tmp[2] * sin(*th);

	/*Rotate vector over angle tilt around y-axis*/
	rotate_coord(X_tmp, tilt);

	/*Tranform vector back to coordinate basis (r, th, phi) at pos_newition (pos_new[1], pos_new[2], pos_new[3])*/
	X[1] = pow(log(pos_new[1] - RB), 1. / RADEXP);
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
	invert_matrix(bl_gcov, bl_gcon);

	dxdxp_func(X, dxdxp);
	invert_matrix(dxdxp, dxpdx);

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

void calc_source(){
	int i, j, z, k, n;
	double a_radius, b_radius, epsilon;
	struct of_geom geom;
	struct of_state q;
	double p_source[NPR], U_s[NPR], om_kepler, r, th, phi, X[NDIM];
	double velocity_factor = 0.7;

	sourceflag = 1;
	epsilon = sqrt(1 + 2.*(0.5*pow(velocity_factor, 2.) - 1.)*pow(velocity_factor, 2.));
	a_radius = rmax / (1. + epsilon);
	b_radius = a_radius*(1. - epsilon);
	period_max = sqrt(pow(a_radius, 3)*4.*pow(M_PI, 2.));
	fprintf(stderr, "Orbital parameters of eccentric orbit are e=%f a=%f p=%f \n", epsilon, a_radius, period_max);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, 0, 0) {
			for (k = 0; k < B1; k++){
				p_source[k] = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];
			}
			p_source[U3] *= velocity_factor;
			p_source[B1] = 0.;
			p_source[B2] = 0.;
			p_source[B3] = 0.;
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			get_state(p_source, &geom, &q);
			primtoflux(p_source, &q, 0, &geom, U_s);

			//Calculate keplerian rotation rate
			om_kepler = 1. / (pow(a_radius, 3. / 2.) + a);

			//Define source term as the value at the apogee/ascending node divided by the orbital rotation frequency
			for (k = 0; k < NPR; k++){
				if (p_source[RHO] > pow(10., -2.)){
					dU_s[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][k] = U_s[k] * om_kepler / (2.*M_PI);
				}
				else{
					dU_s[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][k] = 0.;
				}
			}
		}
	}

	//Reset the grid to floored values
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.e-7*RHOMIN;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.e-7*UUMIN;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
			//coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], n_ord[n], i, j, z);
		}
	}
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
}

/////////////////////
//magnetic field geometry and normalization
#define NORMALFIELD (0)
#define MADFIELD (1)
#define SEMIMAD (2)
#define TOROIDALFIELD (3)

#define WHICHFIELD TOROIDALFIELD

#define NORMALIZE_FIELD_BY_MAX_RATIO (1)
#define NORMALIZE_FIELD_BY_BETAMIN (2)
#define WHICH_FIELD_NORMALIZATION NORMALIZE_FIELD_BY_BETAMIN
//end magnetic field
//////////////////////

//////////////////////
//torus density normalization
#define THINTORUS_NORMALIZE_DENSITY (1)
#define DOAUTOCOMPUTEENK0 (1)

#define NORMALIZE_BY_TORUS_MASS (1)
#define NORMALIZE_BY_DENSITY_MAX (2)

#if(DONUCLEAR)
#define DENSITY_NORMALIZATION NORMALIZE_BY_TORUS_MASS
#else
#define DENSITY_NORMALIZATION NORMALIZE_BY_DENSITY_MAX
#endif
//torus density normalization
//////////////////////

void init_torus_grb(){
	double compute_udt(double r, double th, double a, double l);
	double compute_uuphi(double r, double th, double a, double l);
	double compute_omega(double r, double th, double a, double l);

	int n, i, j, z;
	double r, th, phi, sth, cth;
	double ur, uh, up, u, rho;
	double X[NDIM];
	struct of_geom geom;

	/* for disk interior */
	double l, rin, lnh, expm2chi, up1;
	double DD, AA, SS, thin, sthin, cthin, DDin, AAin, SSin;
	double kappa, hm1;

	/* for magnetic field */
	double rho_av, umax, beta, bsq_ij, bsq_max, norm, q, beta_act;
	double rmax, lfish_calc(double rmax);

	int iglob, jglob, kglob;
	double rancval;
	double omk, lk, kk, c, ang, al, lin, utin, udt, flin, rho_scale_factor, f, hh, eps, rhofloor, ufloor, rho_at_pmax, rhomax, rhoscal, uuscal;
	double torus_mass;

	double amax;

	double Amin, Amax, cutoff_frac = 0.001;

  const double frac_pert = 5.e-2; //increase the perturbation amplitude to 5% to match Sasha's toroidal field setup

	/* radial distribution of angular momentum */
	ang = 0.25;  // = 0 constant ang. mom. torus; 0.25 standard setting for large MAD torii

	/* disk parameters (use fishbone.m to select new solutions) */
	if (WHICHFIELD == MADFIELD){
		rin = 15.;
		rmax = 34.0;
		kappa = 1.e-3;
		beta = 100.;
	}
	else if (WHICHFIELD == TOROIDALFIELD){
		rin = 6.;
		rmax = 13.792;
		kappa = 1.e-2;
		beta = 5.;
	}

	coord(0, 5, 0, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);
	if (rank == 0) {
		fprintf(stderr, "r[5]: %g\n", r);
		fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - a*a)));
		if (r > 1. + sqrt(1. - a*a)) {
			fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n");
		}
		else {
			fprintf(stderr, "\n");
		}
	}

	/* output choices */
	tf = 25000.0;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	rhomax = 0.;
	umax = 0.;
	torus_mass = 0.;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			rancval = ranc(0);

			///
			/// Computations at pressure max
			///

			r = rmax;
			th = M_PI_2;
			omk = 1. / (pow(rmax, 1.5) + a);
			lk = compute_l_from_omega(r, th, a, omk);
			//log(omk) == (2/ang) log(c) + (1 - 2./ang) log(lk) <-- solve for c:
			c = pow(lk, 1 - ang / 2.) * pow(omk, ang / 2.);

			if (0.0 != ang) {
				//variable l case
				kk = pow(c, 2. / ang);
				al = (ang - 2.) / ang;


				///
				/// Computations at torus inner edge
				///

				//l = lin at inner edge, r = rin
				r = rin;
				th = M_PI_2;
				lin = thintorus_findl(r, th, a, c, al);

				//finding DHK03 lin, utin, f (lin)
				utin = compute_udt(r, th, a, lin);
				flin = pow(fabs(1 - kk*pow(lin, 1 + al)), pow(1 + al, -1));


				///
				/// EXTRA CALCS AT PRESSURE MAX TO NORMALIZE DENSITY
				///
				//l at pr. max r, th
#if( THINTORUS_NORMALIZE_DENSITY )
				r = rmax;
				th = M_PI_2;
				l = lk;
				udt = compute_udt(r, th, a, l);
				f = pow(fabs(1 - kk*pow(l, 1 + al)), pow(1 + al, -1));
				hh = flin*utin*pow(f, -1)*pow(udt, -1);
				eps = (-1 + hh)*pow(gam, -1);
				rho_at_pmax = pow((-1 + gam)*eps*pow(kappa, -1), pow(-1 + gam, -1));
				rho_scale_factor = 1.0 / rho_at_pmax;
#if( DOAUTOCOMPUTEENK0 )
				//will be recomputed for every (ti,tj,tk), but is same for all of them, so ok
				global_kappa = kappa * pow(rho_scale_factor, 1 - gam);
#endif    
#else
				rho_scale_factor = 1.0;
#endif

				///
				/// Computations at current point: r, th
				///
				coord(n_ord[n], i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);

				//l at current r, th
				if (r >= rin) {
					l = thintorus_findl(r, th, a, c, al);
					udt = compute_udt(r, th, a, l);
					f = pow(fabs(1 - kk*pow(l, 1 + al)), pow(1 + al, -1));
					hh = flin*utin*pow(f, -1)*pow(udt, -1);
				}
				else {
					l = udt = f = hh = 0.;
				}
			}
			else{
				//l = constant case
				l = c;
				///
				/// Computations at torus inner edge
				///
				//l = lin at inner edge, r = rin
				r = rin;
				th = M_PI_2;
				lin = l; //constant ang. mom.

				//finding DHK03 lin, utin, f (lin)
				utin = compute_udt(r, th, a, lin);
				flin = 1.;  //f(l) is unity everywhere according to Chakrabarti

				///
				/// EXTRA CALCS AT PRESSURE MAX TO NORMALIZE DENSITY
				///
				//l at pr. max r, th
#if( THINTORUS_NORMALIZE_DENSITY )
				r = rmax;
				th = M_PI_2;
				l = lk;
				udt = compute_udt(r, th, a, l);
				f = 1.;
				hh = flin*utin*pow(f, -1)*pow(udt, -1);
				eps = (-1 + hh)*pow(gam, -1);
				rho_at_pmax = pow((-1 + gam)*eps*pow(kappa, -1), pow(-1 + gam, -1));
				rho_scale_factor = 1.0 / rho_at_pmax;
#if( DOAUTOCOMPUTEENK0 )
				//will be recomputed for every (ti,tj,tk), but is same for all of them, so ok
				global_kappa = kappa * pow(rho_scale_factor, 1 - gam);
#endif 
#else
				rho_scale_factor = 1.0;
#endif

				///
				/// Computations at current point: r, th
				///
				coord(n_ord[n], i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				if (r >= rin) {
					udt = compute_udt(r, th, a, l);
					f = 1.;
					hh = utin / udt;
				}
				else {
					hh = 0;
				}
			}
			eps = (-1 + hh)*pow(gam, -1);
			rho = pow((-1 + gam)*eps*pow(kappa, -1), pow(-1 + gam, -1));

			//compute atmospheric values
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			get_rho_u_floor(r, th, phi, &rhofloor, &ufloor);

			/* regions outside torus */
			if (r < rin || isnan(eps) || eps < 0 || rho*rho_scale_factor < rhofloor) {
				/* these values are demonstrably physical
				for all values of a and r */
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
			}
			/* region inside magnetized torus; u^i is calculated in
			* Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			* so it needs to be transformed at the end */
			else {
				u = kappa * pow(rho, gam) / (gam - 1.);

				rho *= rho_scale_factor;
				u *= rho_scale_factor;

				ur = 0.;
				uh = 0.;
				up = compute_uuphi(r, th, a, l);

				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				if (rho > rhomax) rhomax = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u*(1. + frac_pert*(rancval - 0.5));
				if (u > umax && r > rin) umax = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;

				/* convert from 4-vel in BL coords to relative 4-vel in code coords */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);

				//add up mass to compute total torus mass
				torus_mass += gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT] * rho * dV;
			}
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
#if(STAGGERED)
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
#endif
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &torus_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0) fprintf(stderr, "Before normalization: rhomax: %g, torus_mass: %g\n", rhomax, torus_mass);
	if (DENSITY_NORMALIZATION == NORMALIZE_BY_DENSITY_MAX) {
		rho_scale_factor = 1. / rhomax;
		if (rank == 0) fprintf(stderr, "Normalizing by rhomax = 1:\n");
	}
	else if (DENSITY_NORMALIZATION == NORMALIZE_BY_TORUS_MASS) {
		//a factor of fracphi accounts for missing mass outside the wedge
		rho_scale_factor = 0.01 / torus_mass;
		if (rank == 0) fprintf(stderr, "Normalizing by torus_mass = 0.01:\n");
	}
	torus_mass = 0.;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] *= rho_scale_factor;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] *= rho_scale_factor;
			//add up mass to compute total torus mass
			torus_mass += gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT] * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * dV;
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &torus_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	umax *= rho_scale_factor;
	rhomax *= rho_scale_factor;
#if( DOAUTOCOMPUTEENK0 )
  //recompute once again in case normalization changed things
  global_kappa *= pow(rho_scale_factor, 1 - gam);
#endif

	if (rank == 0) fprintf(stderr, "After normalization: rhomax: %g, torus_mass: %g\n", rhomax, torus_mass);

	if (WHICHFIELD == NORMALFIELD) aphipow = 0.;
	else if (WHICHFIELD == MADFIELD || WHICHFIELD == SEMIMAD) aphipow = 2.5 / (3.*(gam - 1.));
	else if (WHICHFIELD == TOROIDALFIELD) aphipow = 2.5 / (3.*(gam - 1.));
	else {
		fprintf(stderr, "Unknown field type: %d\n", (int)WHICHFIELD);
		exit(321);
	}

	//need to bound density before computing vector potential
	bound_prim(p, 1);

	// first find corner-centered vector potential
	for (n = 0; n < n_active; n++) ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3) {
		dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
		dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
		dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
		dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
	}
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3){
			//cannot use get_phys_coords() here because it can only provide coords at CENT
			coord(n_ord[n], i, j, z, CORN, X);
			bl_coord(X, &r, &th, &phi);
			rho_av = 0.25*(
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] +
				p[nl[n_ord[n]]][index_3D(n_ord[n], i - 1, j, z)][RHO] +
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j - 1, z)][RHO] +
				p[nl[n_ord[n]]][index_3D(n_ord[n], i - 1, j - 1, z)][RHO]);
			q = pow(r, aphipow) * rho_av / rhomax;
			if (WHICHFIELD == NORMALFIELD) q -= 0.2;
			if (WHICHFIELD == MADFIELD || WHICHFIELD == SEMIMAD) q = q*q;
			if (q > 0.) dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = q;  //cos(th) gives two loops.
		}
	}

	//need to apply the floor on density to avoid beta ~ ug/(bsq+SMALL) = 0 outside torus when normalizing B
	for (n = 0; n<n_active; n++) fixup(p, n_ord[n]);

	// now differentiate to find cell-centered B, and begin normalization 
	bsq_max = compute_B_from_A();

	if (WHICHFIELD == NORMALFIELD || WHICHFIELD == SEMIMAD){
		if (rank == 0) fprintf(stderr, "initial bsq_max: %g\n", bsq_max);

		//finally, normalize to set field strength 
		beta_act = (gam - 1.)*umax / (0.5*bsq_max);

		if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);

		if (WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_BETAMIN){
			beta_act = normalize_B_by_beta(beta, rmax, &norm);
			if (rank == 0) fprintf(stderr, "Minimum beta = %g, beta = %g\n", beta_act, beta);
		}
		else if (WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_MAX_RATIO) {
			beta_act = normalize_B_by_maxima_ratio(beta, &norm);
			if (rank == 0) fprintf(stderr, "max(pgas)/mas(pmag) = %g\n", beta_act);
		}
		else {
			if (rank == 0) {
				fprintf(stderr, "Unknown magnetic field normalization %d\n", WHICH_FIELD_NORMALIZATION);
				MPI_Finalize();
				exit(2345);
			}
		}
	}
	else if (WHICHFIELD == MADFIELD){
		getmax_densities(p, &rhomax, &umax);

		amax = get_maxprimvalrpow(p, aphipow, RHO);
		if (rank == 0) fprintf(stderr, "amax = %g\n", amax);

		//by now have the fields computed from vector potential

		//here:
		//1) computing bsq
		//2) rescaling field components such that beta = p_g/p_mag is what I want
		//   (constant in the main disk body and tapered off to zero near torus edges)
		normalize_field_local_nodivb(beta, rhomax, amax, p, dq, 1);

		//3) re-compute vector potential by integrating up \int B^r dA in theta
		//   (this uses MPI and zeros out B[3] because B[3] is used to communicate the integration results)
		compute_vpot_from_gdetB1(p, dq);

		Amax = compute_Amax(dq);
		Amin = cutoff_frac * Amax;

		//chop off magnetic field close to the torus boundaries
		for (n = 0; n < n_active; n++){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1 + D1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1 + D2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				if (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] < Amin) {
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = Amin;
				}
			}
		}

		//4) recompute the fields by converting the new A to B
		compute_B_from_A();

		//5) normalize the field
		if (WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_BETAMIN){
			beta_act = normalize_B_by_beta(beta, rmax, &norm);
			if (rank == 0) fprintf(stderr, "Minimum beta = %g, beta = %g\n", beta_act, beta);
		}
		else if (WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_MAX_RATIO){
			beta_act = normalize_B_by_maxima_ratio(beta, &norm);
			if (rank == 0) fprintf(stderr, "max(pgas)/mas(pmag) = %g\n", beta_act);
		}
		else{
			if (rank == 0){
				fprintf(stderr, "Unknown magnetic field normalization %d\n", WHICH_FIELD_NORMALIZATION);
				MPI_Finalize();
				exit(2345);
			}
		}
	}
	else if (WHICHFIELD == TOROIDALFIELD){
		set_uniform_Bphi();
		getmax_densities(p, &rhomax, &umax);
		amax = get_maxprimvalrpow(p, aphipow, RHO);
		if (rank == 0) fprintf(stderr, "amax = %g\n", amax);
		normalize_field_local_nodivb(beta, rhomax, amax, p, dq, 3);
	}

	// enforce boundary conditions
	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);
	bound_prim(p, 1);

#if( DO_FONT_FIX )
	set_Katm();
#endif 

#if (GPU_ENABLED)
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
#endif
}

//note that only axisymmetric A is supported
double compute_Amax(double(*restrict A[NB])[NPR]){
	double Amax = 0.;
	int n, i, j, z;
	struct of_geom geom;

	for (n = 0; n<n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1 + D1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1 + D2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			if (A[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] > Amax) Amax = A[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &Amax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	return(Amax);
}

//note that only axisymmetric A is supported
double compute_B_from_A(void){
	double bsq_max = 0., bsq_ij;
	int n, i, j, z;
	struct of_geom geom;
	#if(TRANS_BOUND && STAGGERED)
	gpu = 0;
	E_average();
	#endif
	for (n = 0; n < n_active; n++){
		#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
			}
		}
		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.;
			}
		}

		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
			get_geometry(n_ord[n], i, j, z, FACE1, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = -(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][3]) / (dx[nl[n_ord[n]]][2] * geom.g)
				#if(N3G>0)
				+ (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][2]) / (dx[nl[n_ord[n]]][3] * geom.g)
				#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE2, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][3]) / (dx[nl[n_ord[n]]][1] * geom.g)
				#if(N3G>0)
				- (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][1]) / (dx[nl[n_ord[n]]][3] * geom.g)
				#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE3, &geom);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = -(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][2]) / (dx[nl[n_ord[n]]][1] * geom.g)
				+ (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][1]) / (dx[nl[n_ord[n]]][2] * geom.g);
		}
		#endif
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] =
				-(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][3]
				+ dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[nl[n_ord[n]]][2] * geom.g)
				+ (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][2]
				+ dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] =
				(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][3]
				- dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[nl[n_ord[n]]][1] * geom.g)
				- (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][1]
				- dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][1] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] =
				-(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][2]
				- dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[nl[n_ord[n]]][1] * geom.g)
				+ (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][1]
				- dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][1] - dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[nl[n_ord[n]]][2] * geom.g);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][1]) / (2.0);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][2]) / (2.0);
			#if(N3G>0)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][3]) / (2.0);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			#endif
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#endif
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			if (bsq_ij > bsq_max && (j != 0 && j != N2 - 1)) bsq_max = bsq_ij;
		}
	}

#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#endif

	return(bsq_max);
}

double normalize_B_by_maxima_ratio(double beta_target, double *norm_value){
	double beta_act, bsq_ij, u_ij, umax = 0., bsq_max = 0.;
	double norm;
	int n, i, j, z;
	struct of_geom geom;

	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			if (bsq_ij > bsq_max && (j != 0 && j != N2 - 1)) bsq_max = bsq_ij;
			u_ij = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			if (u_ij > umax) umax = u_ij;
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	/* finally, normalize to set field strength */
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);

	norm = sqrt(beta_act / beta_target);
	if (norm_value) *norm_value = norm;

	bsq_max = 0.;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1 + D1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1 + D2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= norm;
			#if(STAGGERED)
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= norm;
			#endif
		}
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			if (bsq_ij > bsq_max && (j != 0 && j != N2 - 1)) bsq_max = bsq_ij;
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);

	return(beta_act);
}

//normalize the magnetic field using the values inside r < rmax
double normalize_B_by_beta(double beta_target, double rmax, double *norm_value){
	double beta_min = 1e100, beta_ij, beta_act, bsq_ij, u_ij, umax = 0., bsq_max = 0.;
	double norm;
	int n, i, j, z;
	struct of_geom geom;
	double X[NDIM], r, th, ph;

	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &ph);
			if (r > rmax) continue;

			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			u_ij = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			beta_ij = (gam - 1.)*u_ij / (0.5*(bsq_ij + SMALL));
			if (beta_ij < beta_min && (j != 0 && j != N2 - 1)) beta_min = beta_ij;
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &beta_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	/* finally, normalize to set field strength */
	beta_act = beta_min;
	norm = sqrt(beta_act / beta_target);
	if (norm_value) *norm_value = norm;

	beta_min = 1e100;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1 + D1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1 + D2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= norm;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= norm;
			#if(STAGGERED)
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= norm;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= norm;
			#endif
		}
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			u_ij = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			beta_ij = (gam - 1.)*u_ij / (0.5*(bsq_ij + SMALL));
			if (beta_ij < beta_min && (j != 0 && j != N2 - 1)) beta_min = beta_ij;
		}
	}

	//exchange the info between the MPI processes to get the true max
	MPI_Allreduce(MPI_IN_PLACE, &beta_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	beta_act = beta_min;

	return(beta_act);
}

/////////////////////////////////////////////////////////////////
//
// init_torus_grb() preliminaries
//
/////////////////////////////////////////////////////////////////
#define JMAX 100

double rtbis(double(*func)(double, double*), double *parms, double x1, double x2, double xacc){
	//Using bisection, find the root of a function func known to lie between x1 and x2. The root,
	//returned as rtbis, will be refined until its accuracy is \pm xacc.
	//taken from http://gpiserver.dcom.upv.es/Numerical_Recipes/bookcpdf/c9-1.pdf
	int j;
	double dx, f, fmid, xmid, rtb;
	f = (*func)(x1, parms);
	fmid = (*func)(x2, parms);
	if (f*fmid >= 0.0) {
		fprintf(stderr, "f(%g)=%g f(%g)=%g\n", x1, f, x2, fmid);
		fprintf(stderr, "Root must be bracketed for bisection in rtbis\n");
		exit(434);
	}
	rtb = (f < 0.0) ? (dx = x2 - x1, x1) : (dx = x1 - x2, x2); //Orient the search so that f>0 lies at x+dx.
	for (j = 1; j <= JMAX; j++) {
		fmid = (*func)(xmid = rtb + (dx *= 0.5), parms); //Bisection loop.
		if (fmid <= 0.0) {
			rtb = xmid;
		}
		if (fabs(dx) < xacc || fmid == 0.0) {
			return rtb;
		}
	}
	fprintf(stderr, "Too many bisections in rtbis");
	return 0.0; //Never get here.
}

double lfunc(double lin, double *parms){
	double gutt, gutp, gupp, al, c;
	double ans;

	gutt = parms[0];
	gutp = parms[1];
	gupp = parms[2];
	al = parms[3];
	c = parms[4];

	ans = (gutp - lin * gupp) / (gutt - lin * gutp) - c *pow(lin / c, al); // (lin/c) form avoids catastrophic cancellation due to al = 2/n - 1 >> 1 for 2-n << 1

	return(ans);
}

void compute_gu(double r, double th, double a, double *gutt, double *gutp, double *gupp){
	//metric (expressions taken from eqtorus_c.nb):
	*gutt = -1 - 4 * r*(pow(a, 2) + pow(r, 2))*
		pow((-2 + r)*r + pow(a, 2), -1)*
		pow(pow(a, 2) + cos(2 * th)*pow(a, 2) + 2 * pow(r, 2), -1);

	*gutp = -4 * a*r*pow((-2 + r)*r + pow(a, 2), -1)*
		pow(pow(a, 2) + cos(2 * th)*pow(a, 2) + 2 * pow(r, 2), -1);

	*gupp = 2 * ((-2 + r)*r + pow(a, 2)*pow(cos(th), 2))*
		pow(sin(th), -2)*pow((-2 + r)*r + pow(a, 2), -1)*
		pow(pow(a, 2) + cos(2 * th)*pow(a, 2) + 2 * pow(r, 2), -1);
}

double thintorus_findl(double r, double th, double a, double c, double al){
	double gutt, gutp, gupp;
	double parms[5];
	double l;

	compute_gu(r, th, a, &gutt, &gutp, &gupp);

	//store params in an array before function call
	parms[0] = gutt;
	parms[1] = gutp;
	parms[2] = gupp;
	parms[3] = al;
	parms[4] = c;

	//solve for lin using bisection, specify large enough root search range, (1e-3, 1e3)
	//demand accuracy 5x machine prec.
	//in non-rel limit l_K = sqrt(r), use 10x that as the upper limit:
	l = rtbis(&lfunc, parms, 1, 10 * sqrt(r), 5.*DBL_EPSILON);

	return(l);
}

double compute_udt(double r, double th, double a, double l){
	double gutt, gutp, gupp;
	double udt;

	compute_gu(r, th, a, &gutt, &gutp, &gupp);

	udt = -sqrt(-1 / (gutt - 2 * l * gutp + l * l * gupp));

	return(udt);
}

double compute_omega(double r, double th, double a, double l){
	double gutt, gutp, gupp;
	double omega1;

	compute_gu(r, th, a, &gutt, &gutp, &gupp);

	omega1 = (gutp - gupp*l)*pow(gutt - gutp*l, -1);

	return(omega1);
}

double compute_uuphi(double r, double th, double a, double l){
	double gutt, gutp, gupp;
	double udt, udphi, uuphi;

	//u_t
	udt = compute_udt(r, th, a, l);

	//u_phi
	udphi = -udt * l;

	compute_gu(r, th, a, &gutt, &gutp, &gupp);

	//u^phi
	uuphi = gutp * udt + gupp * udphi;
	return(uuphi);
}

double compute_l_from_omega(double r, double th, double a, double omega1){
	double gutt, gutp, gupp;
	double l;

	compute_gu(r, th, a, &gutt, &gutp, &gupp);
	l = (gutp - omega1 * gutt) / (gupp - omega1 * gutp);

	return(l);
}


void getmax_densities(double(*restrict prim[NB])[NPR], double *rhomax, double *umax){
	int n, i, j, z;

	*rhomax = 0;
	*umax = 0;
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			if (prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] > *rhomax)   *rhomax = prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO];
			if (prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > *umax)    *umax = prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, rhomax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

double get_maxprimvalrpow(double(*restrict prim[NB])[NPR], double rpow, int m){
	int n, i, j, z;
	double X[NDIM];
	double  r, th, ph;

	double val;
	double maxval = -DBL_MAX;

	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &ph);

			val = pow(r, rpow)*prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][m];
			if (val > maxval) maxval = val;
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	return(maxval);
}

int normalize_field_local_nodivb(double targbeta, double rhomax, double amax, double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR], int dir){
	int n, i, j, z;
	double ratc_ij;

	bound_prim(prim, 1);
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			//cell centered ratio in this cell
			ratc_ij = compute_rat(prim, A, rhomax, amax, targbeta, CENT, n_ord[n], i, j, z);

			// normalize staggered field primitive
			if (dir == 1) prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= ratc_ij;
			if (dir == 2) prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= ratc_ij;
			if (dir == 3) prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= ratc_ij;
		}
	}
	return(0);
}

#define MYSMALL (1.e-300)

//Returns: factor to multiply field components by to get the desired
//value of beta: targbeta = p_g/p_mag
double compute_rat(double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR], double rhomax, double amax, double targbeta, int loc, int n, int i, int j, int z){
	double bsq_ij, pg_ij, beta_ij, rat_ij;
	struct of_geom geom;
	double X[NDIM];
	double rat, ratc;
	double profile;
	double  r, th, ph;
	double rho, u;

// copied example from elsewhere:
//  get_geometry(n_ord[n], i, j, z, CENT, &geom);
//  bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
//  u_ij = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
//  beta_ij = (gam - 1.)*u_ij / (0.5*(bsq_ij + SMALL));

  get_geometry(n, i, j, z, loc, &geom);
	//coord(n, i, j, z, loc, X);
	//bl_coord(X, &r, &th, &ph);

	bsq_ij = bsq_calc(prim[nl[n]][index_3D(n, i, j, z)], &geom);

	rho = prim[nl[n]][index_3D(n, i, j, z)][RHO];
	//use the following instead of MACP0A1(prim,i,j,k,UU) because the latter
	//can be perturbed by random noise, which we want to avoid
	u = global_kappa * pow(rho, gam) / (gam - 1.);

	//EOSMARK
	pg_ij = (gam - 1)*u;
	beta_ij = 2 * pg_ij / (bsq_ij + MYSMALL);
	rat_ij = sqrt(beta_ij / targbeta); //ratio at CENT

	//rescale rat_ij so that:
	// rat_ij = 1 inside the main body of torus
	// rat_ij = 0 outside of main body of torus
	// rat_ij ~ rho in between
	//ASSUMING DENSITY HAS ALREADY BEEN NORMALIZED -- SASMARK
	profile = compute_profile(prim, amax, aphipow, loc, n, i, j, z);
	rat_ij *= profile;

	return(rat_ij);
}


double compute_profile(double(*restrict prim[NB])[NPR], double amax, double aphipow, int loc, int n, int i, int j, int z){
	double X[NDIM], r, th, ph;
	struct of_geom geom;
	double profile;

	get_geometry(n, i, j, z, loc, &geom);
	coord(n, i, j, z, loc, X);
	bl_coord(X, &r, &th, &ph);

	profile = (log10(pow(r, aphipow)*prim[nl[n]][index_3D(n, i, j, z)][RHO] / amax + SMALL) + 3.) / 1.0;
	if (profile<0.) profile = 0.;
	if (profile>1.) profile = 1.;
	//profile = 1.; //SashaTch commented out this line because want zero field outside torus
	return(profile);
}

//compute vector potential assuming B_\phi = 0 and zero flux at poles
//(not tested in non-axisymmetric field distribution but in principle should work)
int compute_vpot_from_gdetB1(double(*restrict prim[NB])[NPR], double(*restrict A[NB])[NPR]){
	int n, i, j, z;
	int jj;
	int ci, cj, cz;
	int dj, js, je, jsb, jeb;
	struct of_geom geom;
	double gdet;
	int finalstep;

	//first, bound to ensure consistency of magnetic fields across tiles
	bound_prim(prim, 1);

	if (NB_2 == 1) {
		//1-cpu version
		for (n = 0; n < n_active; n++){
			for (i = N1_GPU_offset[n_ord[n]]; i < N1_GPU_offset[n_ord[n]] + BS_1 + D1; i++) {
				for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3 + D3; z++) {
					//zero out starting element of vpot
					A[nl[n_ord[n]]][index_3D(n_ord[n], i, 0, z)][3] = 0.0;
					//integrate vpot along the theta line
					for (j = N2_GPU_offset[n_ord[n]]; j < N2_GPU_offset[n_ord[n]] + BS_2 / 2; j++) {
						get_geometry(n_ord[n], i, j, z, CENT, &geom);
						gdet = geom.g;

						//take a loop along j-line at a fixed i,k and integrate up vpot
						A[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][3] = A[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] * gdet*dx[nl[n_ord[n]]][2];
					}
					A[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.0;
					//integrate vpot along the theta line
					for (j = N2_GPU_offset[n_ord[n]] + BS_2; j > N2_GPU_offset[n_ord[n]] + BS_2 / 2; j--) {
						get_geometry(n_ord[n], i, j - 1, z, CENT, &geom);
						gdet = geom.g;

						//take a loop along j-line at a fixed i,k and integrate up vpot
						A[nl[n_ord[n]]][index_3D(n_ord[n], i, j - 1, z)][3] = A[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j - 1, z)][B1] * gdet*dx[nl[n_ord[n]]][2];
					}
				}
			}
		}
	}
	else {
		//Loop over all zero-level blocks
		for (cj = 0; cj < NB_2 / 2; cj++) {
			for (ci = 0; ci < NB_1; ci++)for (cz = 0; cz < NB_3; cz++) {
				n = AMR_coord_linear(0, ci, cj, cz);
				dj = 1;
				js = N2_GPU_offset[n];
				jsb = N2_GPU_offset[n];
				je = N2_GPU_offset[n] + BS_2;
				jeb = N2_GPU_offset[n] + BS_2 - 1;
				//then it's the turn of the current row of CPUs to pick up where the previous row has left it off
				//since pstag is bounded unlike A, use pstag[B3] as temporary space to trasnfer values of A[3] between CPUs
				//initialize lowest row of A[3]
				if (block[n][AMR_NODE] == rank) {
					for (i = N1_GPU_offset[n]; i < N1_GPU_offset[n] + BS_1 + D1; i++) {
						for (z = N3_GPU_offset[n]; z < N3_GPU_offset[n] + BS_3 + D3; z++) {
							//zero out or copy starting element of vpot
							if (0 == cj) {
								//if CPU is at physical boundary, initialize (zero out) A[3]
								A[n][index_3D(n, i, js, z)][3] = 0.0;
							}
							else {
								//else copy B[3] (which was bounded below) -> A[3]
								A[n][index_3D(n, i, js, z)][3] = prim[n][index_3D(n, i, jsb - dj, z)][B3];
							}
							//integrate vpot along the theta line
							for (j = js; j != je; j += dj) {
								get_geometry(n, i, j - js + jsb, z, CENT, &geom);
								gdet = geom.g;
								//take a loop along j-line at a fixed i,k and integrate up vpot
								A[n][index_3D(n, i, j + dj, z)][3] = A[n][index_3D(n, i, j, z)][3] + dj * prim[n][index_3D(n, i, j - js + jsb, z)][B1] * gdet*dx[n][2];
							}
							//copy A[3] -> B[3] before bounding
							prim[n][index_3D(n, i, jeb, z)][B3] = A[n][index_3D(n, i, je, z)][3];
						}
					}
				}

				n = AMR_coord_linear(0, ci, NB_2 - cj - 1, cz);
				dj = -1;
				js = N2_GPU_offset[n] + BS_2;
				jsb = N2_GPU_offset[n] + BS_2 - 1;
				je = N2_GPU_offset[n];
				jeb = N2_GPU_offset[n];
				//then it's the turn of the current row of CPUs to pick up where the previous row has left it off
				//since pstag is bounded unlike A, use pstag[B3] as temporary space to trasnfer values of A[3] between CPUs
				//initialize lowest row of A[3]
				if (block[n][AMR_NODE] == rank) {
					for (i = N1_GPU_offset[n]; i < N1_GPU_offset[n] + BS_1 + D1; i++) {
						for (z = N3_GPU_offset[n]; z < N3_GPU_offset[n] + BS_3 + D3; z++) {
							//zero out or copy starting element of vpot
							if (0 == cj) {
								//if CPU is at physical boundary, initialize (zero out) A[3]
								A[n][index_3D(n, i, js, z)][3] = 0.0;
							}
							else {
								//else copy B[3] (which was bounded below) -> A[3]
								A[n][index_3D(n, i, js, z)][3] = prim[n][index_3D(n, i, jsb - dj, z)][B3];
							}
							//integrate vpot along the theta line
							for (j = js; j != je; j += dj) {
								get_geometry(n, i, j - js + jsb, z, CENT, &geom);
								gdet = geom.g;
								//take a loop along j-line at a fixed i,k and integrate up vpot
								A[n][index_3D(n, i, j + dj, z)][3] = A[n][index_3D(n, i, j, z)][3] + dj * prim[n][index_3D(n, i, j - js + jsb, z)][B1] * gdet*dx[n][2];
							}
							//copy A[3] -> B[3] before bounding
							prim[n][index_3D(n, i, jeb, z)][B3] = A[n][index_3D(n, i, je, z)][3];
						}
					}
				}
			}
			//just in case, wait until all CPUs get here
			MPI_Barrier(MPI_COMM_WORLD);
			//bound here
			bound_prim(prim, 1);
		}
	}

	//ensure consistency of vpot across the midplane
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_COORD2] == NB_2 / 2) {
			for (i = N1_GPU_offset[n_ord[n]]; i < N1_GPU_offset[n_ord[n]] + BS_1 + D1; i++) {
				for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3 + D3; z++) {
					A[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = prim[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] - 1, z)][B3];
				}
			}
		}
	}

	//need to zero out prim[B3] everywhere
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1 + D1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1 + D2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			prim[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
		}
	}
	return(0);
}

#define BL (1)

void get_rho_u_floor(double r, double th, double phi, double *rho_floor, double *u_floor)
{
	double rhoflr, uuflr;
	double uuscal, rhoscal;


	if (BL == 0){
		rhoflr = 1e-6;
		uuflr = pow(rhoflr, gam);
	}
	else{
		rhoscal = pow(r, -POWRHO);
		uuscal = pow(rhoscal, gam); //rhoscal/r ;
		rhoflr = RHOMIN*rhoscal; //this is Rodrigo's rhot
		uuflr = UUMIN*uuscal;

		if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
		if (uuflr  < UUMINLIMIT) uuflr = UUMINLIMIT;
	}

	*rho_floor = rhoflr;
	*u_floor = uuflr;
}

//note that only axisymmetric A is supported
void set_uniform_Bphi(void){
	int n, i, j, z;
	struct of_geom geom;
	#if(TRANS_BOUND && STAGGERED)
	gpu = 0;
	E_average();
	#endif
	for (n = 0; n < n_active; n++){
		#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
			}
		}
		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.;
			}
		}

		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3){
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 1.;
		}
		#endif
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 1.;
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][1]) / (2.0);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][2]) / (2.0);
			#if(N3G>0)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][3]) / (2.0);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			#endif
			#endif
		}
	}
	return;
}
