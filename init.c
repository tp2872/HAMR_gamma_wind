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
#include "decs_MPI.h"

void rotate_vector2(double V[NDIM], double pos_new[NDIM], double *r, double *th, double *phi, double tilt);
void coord_transform(double *pr, int n, int ii, int jj, int zz);
void set_mag(void);

typedef struct {
  double xmin, xmax, ymin, ymax, zmin, zmax; //array extent
  int nvars, nx, ny, nz; //resolution
} extent;

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
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, 0, 0) {
			for (k = 0; k < B1; k++){
				p_source[k] = p[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
			}
			p_source[U3] *= velocity_factor;
			p_source[B1] = 0.;
			p_source[B2] = 0.;
			p_source[B3] = 0.;
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			get_state(p_source, &geom, &q);
			primtoflux(p_source, &q, TT, &geom, U_s);

			//Calculate keplerian rotation rate
			om_kepler = 1. / (pow(a_radius, 3. / 2.) + a);

			//Define source term as the value at the apogee/ascending node divided by the orbital rotation frequency
			for (k = 0; k < NPR; k++){
				if (p_source[RHO] > pow(10., -2.)){
					dU_s[n_ord[n]][index2(n_ord[n] ,i, j, z)][k] = U_s[k] * om_kepler / (2.*M_PI);
				}
				else{
					dU_s[n_ord[n]][index2(n_ord[n] ,i, j, z)][k] = 0.;
				}
			}
		}
	}

	//Reset the grid to floored values
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO] = 1.e-7*RHOMIN;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU] = 1.e-7*UUMIN;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][U1] = 0.0;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][U2] = 0.0;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][U3] = 0.0;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B1] = 0.0;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B2] = 0.0;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] = 0.0;
			//coord_transform(p[n_ord[n]][index(n_ord[n] ,i, j, z)],n_ord[n], i, j, z);
		}
	}
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
}

void init()
{
  void init_bondi(void);
  void init_torus(void);
  void init_disruption(void);
  void init_monopole(double Rout_val);

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
  case DISRUPTION_PROBLEM:
	//init_torus();
    init_disruption();
    break;
  case BONDI_PROBLEM_1D:
  case BONDI_PROBLEM_2D:
    init_bondi();
    break;
  }

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
	double lfish_calc(double rmax) ;

	/* disk parameters (use fishbone.m to select new solutions) */
    //a = 0.9375 ;
	//rin = 36.;
	//rmax = 73.9672;
	//rin = 5.*36. ;
	//rmax = 361.95;

	
	//rmax = 73.962 ;
	
	double temp = a;
	a = 0.9375;
	rin = 12.5;
	//rmax = 14.6145;
	rmax = 25.;
	//rmax = 14.6165;
	///rin = 12.;
	//rmax = 14.616;
	//rmax = 24.;
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
	DTd = 25.;	/* dumping frequency, in units of M */
	DTl = 50.0;	/* logfile frequency, in units of M */
	DTi = 100.0; 	/* image file frequ., in units of M */
	DTr = 5.0 * 1000.; 	/* restart file frequ., in timesteps */

	/* start diagnostic counters */
	dump_cnt = 0 ;
	image_cnt = 0 ;
	rdump_cnt = 0 ;
	defcon = 1. ;

	rhomax = 0. ;
	umax = 0. ;
	tilt = TILT_ANGLE/180.*M_PI;
	eccentricity = 0.0;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
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

				p[n_ord[n]][index(n_ord[n] ,i,j,z)][RHO] = rho;
				p[n_ord[n]][index(n_ord[n] ,i,j,z)][UU] = u;
				p[n_ord[n]][index(n_ord[n] ,i,j,z)][U1] = ur;
				p[n_ord[n]][index(n_ord[n] ,i,j,z)][U2] = uh;
				p[n_ord[n]][index(n_ord[n] ,i,j,z)][U3] = up;
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

				p[n_ord[n]][index(n_ord[n] ,i,j,z)][RHO] = rho;
				if(rho > rhomax) rhomax = rho ;
				p[n_ord[n]][index(n_ord[n] ,i,j,z)][UU] = u*(1. + 4.e-2*(ranc(0) - 0.5));
				if(u > umax && r > rin) umax = u ;
			
				#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
				//th = (th - M_PI / 2.) *(fractheta*0.5) + M_PI / 2.;
				rotate_vector(V, pos_new, &r, &th, &phi, tilt);
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U1] = V[1];
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U2] = V[2];
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U3] = V[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[n_ord[n]][index(n_ord[n] ,i,j,z)],n_ord[n], i, j, z);
				#elif(ELLIPTICAL)
				V_old[1] = ur;
				V_old[2] = uh;
				V_old[3] = up;
				elliptical_vector(X_cart, V_old, V_new, pos_new, &r, &th, eccentricity);
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U1] = V_new[1];
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U2] = V_new[2];
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U3] = V_new[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[n_ord[n]][index(n_ord[n] ,i,j,z)],n_ord[n], i, j, z);
				#else
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U1] = ur;
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U2] = uh;
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][U3] = up;//watch out

				/* convert from 4-vel to 3-vel */
				coord_transform(p[n_ord[n]][index(n_ord[n] ,i, j, z)],n_ord[n], i, j, z);
				#endif
			}
			p[n_ord[n]][index(n_ord[n] ,i,j,z)][B1] = 0.;
			p[n_ord[n]][index(n_ord[n] ,i,j,z)][B2] = 0.;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] = 0.;
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
	if (rank == 0){
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	//ZSLOOP(0,N1-1,0,N2-1) {
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO] /= rhomax;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU] /= rhomax;
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

	#if (GPU_ENABLED)
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
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
  double lfish_calc(double rmax) ;
  
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
  //rin = 36.;
  //rmax = 73.9672;
  //rin = 5.*36. ;
  //rmax = 361.95;
  
  
  //rmax = 73.962 ;
  
  rin = 6;
  //rmax = 14.6145;
  rmax = 12.;
  //rmax = 14.6165;
  ///rin = 12.;
  //rmax = 14.616;
  //rmax = 24.;
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
  DTd = 25.0;	/* dumping frequency, in units of M */
  DTl = 50.0;	/* logfile frequency, in units of M */
  DTi = 100.0; 	/* image file frequ., in units of M */
  DTr = 5.0 * 1000.; 	/* restart file frequ., in timesteps */
  
  /* start diagnostic counters */
  dump_cnt = 0 ;
  image_cnt = 0 ;
  rdump_cnt = 0 ;
  defcon = 1. ;

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
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif
  //vars: [x],[y],[z],[rho],[ug],[vx],[vy],[vz],[poten]
  //ivar:  0,  1,  2,   3,   4,   5,   6,   7,     8
  //mapping: icdata[((ivar*nx+ii)*ny+jj)*nz+kk]
  
  rhomax = 0. ;
  umax = 0. ;
  tilt = TILT_ANGLE/180.*M_PI;
  eccentricity = 0.0;
  for (n = 0; n < n_active; n++){
    ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
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
      PLOOP p[n_ord[n]][index(n_ord[n], i, j, z)][k] = prim[k];
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
  MPI_Barrier(mpi_cartcomm);
  MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
  
  /*Share umax among MPI processes*/
  MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
  MPI_Barrier(mpi_cartcomm);
  #endif
  
  /* Normalize the densities so that max(rho) = 1 */
  if (rank == 0){
    fprintf(stderr, "rhomax: %g\n", rhomax);
  }
  //ZSLOOP(0,N1-1,0,N2-1) {
  for (n = 0; n < n_active; n++){
    ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
      p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO] /= rhomax;
      p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU] /= rhomax;
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


	#if (GPU_ENABLED)
  for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
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
  p[UU] = 0.01*p[RHO];

//  p[U1] = (vx * sin(th)*cos(ph) + vy * sin(th)*sin(ph) + vz * cos(th)) / sqrt(bl_gcov[1][1]);
 // p[U2] = (vx * cos(th)*cos(ph) + vy * cos(th)*sin(ph) - vz * sin(th)) / sqrt(bl_gcov[2][2]);
 /// p[U3] = (-vx * sin(ph) + vy * cos(ph)) / sqrt(bl_gcov[3][3]);

 
  p[B1] = 0.;
  p[B2] = 0.;
  p[B3] = 0.;
  if(vx*vx+vy*vy+vz*vz>1.0)printf("test:%f %f %f %f %f %f %f %f %f \n", x1,y1,z1, p[RHO], p[UU],vx,vy,vz, poten);

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
  i1 = ceil(i);
  j1 = ceil(j);
  k1 = ceil(k);
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
	double rhomax = 0., umax = 0.;
	int i100 = 0;
	double rho_av, q, beta = 100., bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM],pos_new[NDIM];
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	double tilt = TILT_ANGLE / 180.*M_PI;
	
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
			if (p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO]> rhomax) rhomax = p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO];
			if (p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU] > umax) umax = p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU];
		}
	}

	#if (MPI_enable)
	/*Share rhomax among MPI processes*/
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);

	/*Share umax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0){
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1) {
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO] /= rhomax;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][UU] /= rhomax;
		}
	}
	umax /= rhomax;
	rhomax = 1.;
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);

	do{
		i100++;
		coord(0, i100, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
	} while (r < 400.0);
	/* first find corner-centered vector potential */
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]]){
			dq[n_ord[n]][index(n_ord[n] ,i, j, z)][0] = 0.;
			dq[n_ord[n]][index(n_ord[n], i, j, z)][1] = 0.;
			dq[n_ord[n]][index(n_ord[n], i, j, z)][2] = 0.;
			dq[n_ord[n]][index(n_ord[n], i, j, z)][3] = 0.;
		}
	}
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]]){
			/* field-in-disk version */
			/* flux_ct */
			rho_av = 0.25*(
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO] +
				p[n_ord[n]][index(n_ord[n] ,i - 1, j, z)][RHO] +
				p[n_ord[n]][index(n_ord[n] ,i, j - 1, z)][RHO] +
				p[n_ord[n]][index(n_ord[n] ,i - 1, j - 1, z)][RHO]);
			//rho_av = p[n_ord[n]][index(n_ord[n] ,i, j, z)][RHO];
			q = rho_av / rhomax-0.2;
			if (q > 0. && i < i100){
				coord(n_ord[n],i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				//dq[n_ord[n]][index(n_ord[n], i, j, z)][2] = q*r*r; //Toroidal
				//dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3] = dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3]* pow(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3], 2.0) * pow(r, 3.0)*sqrt(pow(cos((X[1] - 2.0) * 2.0*M_PI / 1.0), 2.0))*sqrt(pow(cos((X[2] - 0.5) * 2.*M_PI / 0.1), 2.0)) / 10.;
				dq[n_ord[n]][index(n_ord[n], i, j, z)][3] =  pow(q, 2.0) * pow(r, 3.0); //MAD
				//3d jet
				//X[1] = log(r - RB);
				//dq[n_ord[n]][index(n_ord[n], i, j, z)][3] = pow(dq[n_ord[n]][index(n_ord[n], i, j, z)][3], 3.0)* pow(r, 3.0)*(0.1 + 0.9*sqrt(pow(cos((X[1] - 2.0) * 2.0*M_PI / 0.5), 2.0))*sqrt(pow(cos((X[2] - 0.5) * 2.*M_PI / 0.05), 2.0))) / 10;
			}
			else{
				dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3] = 0.0;
			}
			if (q > 0.){
				#if (TILTED)
				coord(n_ord[n],i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				pos_new[1] = r;
				pos_new[2] = th;
				pos_new[3] = phi;
				sph_to_cart(X_cart, &r, &th, &phi);
				rotate_coord(X_cart, -tilt);
				cart_to_sph(X_cart, &r, &th, &phi);
				V[1] = dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1];
				V[2] = dq[n_ord[n]][index(n_ord[n] ,i, j, z)][2];
				V[3] = dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3];
				rotate_vector2(V, pos_new, &r, &th, &phi, tilt);
				//rotate_vector(V, pos_new, &r, &th, &phi, tilt);
				//coord_transform(V,n_ord[n], i, j, z);
				dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1] = V[1];
				dq[n_ord[n]][index(n_ord[n] ,i, j, z)][2] = V[2];
				dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3] = V[3];
				#endif
			}
		}
	}


	/* now differentiate to find cell-centered B,
	and begin normalization */
	double bsq_max = 0.;
	#if(TRANS_BOUND && STAGGERED)
	gpu = 0;
	//E_average();
	#endif
	for (n = 0; n < n_active; n++){
		#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
		//#if(TRANS_BOUND)
		if(block[n_ord[n]][AMR_NBR1]==-1 || block[n_ord[n]][AMR_POLE]==1 || block[n_ord[n]][AMR_POLE]==3 ){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1 + D3){
				dq[n_ord[n]][index(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
			}
		}
		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1 + D3){
				dq[n_ord[n]][index(n_ord[n], i, N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]], z)][3] = 0.;
			}
		}
		//#endif
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] - 1 + D3){
			get_geometry(n_ord[n], i, j, z, FACE1, &geom);
			ps[n_ord[n]][index(n_ord[n], i, j, z)][1] = -(dq[n_ord[n]][index(n_ord[n], i, j, z)][3] - dq[n_ord[n]][index(n_ord[n], i, j + D2, z)][3]) / (dx[n_ord[n]][2] * geom.g)
			#if(N3G>0)
				+ (dq[n_ord[n]][index(n_ord[n], i, j, z)][2] - dq[n_ord[n]][index(n_ord[n], i, j, z + D3)][2]) / (dx[n_ord[n]][3] * geom.g)
			#endif
				;
			get_geometry(n_ord[n],i, j, z, FACE2, &geom);
			ps[n_ord[n]][index(n_ord[n], i, j, z)][2] = (dq[n_ord[n]][index(n_ord[n], i, j, z)][3] - dq[n_ord[n]][index(n_ord[n], i + D1, j, z)][3]) / (dx[n_ord[n]][1] * geom.g)
			#if(N3G>0)
				-(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1] - dq[n_ord[n]][index(n_ord[n] ,i, j, z + D3)][1]) / (dx[n_ord[n]][3] * geom.g)
			#endif
				;
			get_geometry(n_ord[n],i, j, z, FACE3, &geom);
			ps[n_ord[n]][index(n_ord[n] ,i, j, z)][3] = -(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][2] - dq[n_ord[n]][index(n_ord[n] ,i + D1, j, z)][2]) / (dx[n_ord[n]][1] * geom.g)
				+ (dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1] - dq[n_ord[n]][index(n_ord[n] ,i, j + D2, z)][1]) / (dx[n_ord[n]][2] * geom.g);
		}

		#endif
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B1] =
				-(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3] - dq[n_ord[n]][index(n_ord[n] ,i, j + 1, z)][3]
				+ dq[n_ord[n]][index(n_ord[n], i + 1, j, z)][3] - dq[n_ord[n]][index(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[n_ord[n]][2] * geom.g)
				+ (dq[n_ord[n]][index(n_ord[n] ,i, j, z)][2] - dq[n_ord[n]][index(n_ord[n] ,i, j, z + 1)][2]
				+ dq[n_ord[n]][index(n_ord[n], i + 1, j, z)][2] - dq[n_ord[n]][index(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[n_ord[n]][3] * geom.g);
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B2] =
				(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][3] + dq[n_ord[n]][index(n_ord[n] ,i, j + 1, z)][3]
				- dq[n_ord[n]][index(n_ord[n], i + 1, j, z)][3] - dq[n_ord[n]][index(n_ord[n], i + 1, j + 1, z)][3]) / (2.*dx[n_ord[n]][1] * geom.g)
				- (dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1] + dq[n_ord[n]][index(n_ord[n] ,i, j + 1, z)][1]
				- dq[n_ord[n]][index(n_ord[n], i, j, z + 1)][1] - dq[n_ord[n]][index(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[n_ord[n]][3] * geom.g);
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] =
				-(dq[n_ord[n]][index(n_ord[n] ,i, j, z)][2] + dq[n_ord[n]][index(n_ord[n] ,i, j, z + 1)][2]
				- dq[n_ord[n]][index(n_ord[n], i + 1, j, z)][2] - dq[n_ord[n]][index(n_ord[n], i + 1, j, z + 1)][2]) / (2.*dx[n_ord[n]][1] * geom.g)
				+ (dq[n_ord[n]][index(n_ord[n] ,i, j, z)][1] + dq[n_ord[n]][index(n_ord[n] ,i, j, z + 1)][1]
				- dq[n_ord[n]][index(n_ord[n], i, j + 1, z)][1] - dq[n_ord[n]][index(n_ord[n], i, j + 1, z + 1)][1]) / (2.*dx[n_ord[n]][2] * geom.g);
			#else
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B1] = (ps[n_ord[n]][index(n_ord[n] ,i, j, z)][1] + ps[n_ord[n]][index(n_ord[n] ,i + 1, j, z)][1]) / (2.0);
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B2] = (ps[n_ord[n]][index(n_ord[n] ,i, j, z)][2] + ps[n_ord[n]][index(n_ord[n] ,i, j + 1, z)][2]) / (2.0);
			#if(N3G>0)
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] = (ps[n_ord[n]][index(n_ord[n] ,i, j, z)][3] + ps[n_ord[n]][index(n_ord[n] ,i, j, z + 1)][3]) / (2.0);
			#else
			p[n_ord[n]][index(n_ord[n], i, j, z)][B3] = ps[n_ord[n]][index(n_ord[n], i, j, z)][3];
			#endif
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#endif
			bsq_ij = bsq_calc(p[n_ord[n]][index(n_ord[n] ,i, j, z)], &geom);
			if (bsq_ij > bsq_max) bsq_max = bsq_ij;
		}
	}



	#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif

	if (rank == 0){
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}

	/* finally, normalize to set field strength */
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	if (rank == 0){
		fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	}
	norm = sqrt(beta_act / beta);
	bsq_max = 0.;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]]-1+D3){
		//ZLOOP3D_MPI{
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B1] *= norm;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B2] *= norm;
			p[n_ord[n]][index(n_ord[n] ,i, j, z)][B3] *= norm;
			#if(STAGGERED)
			ps[n_ord[n]][index(n_ord[n] ,i, j, z)][1] *= norm;
			ps[n_ord[n]][index(n_ord[n] ,i, j, z)][2] *= norm;
			ps[n_ord[n]][index(n_ord[n] ,i, j, z)][3] *= norm;
			#endif

			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[n_ord[n]][index(n_ord[n] ,i, j, z)], &geom);
			if (bsq_ij > bsq_max) bsq_max = bsq_ij;
		}
	}

	/*Share bsq_max among MPI processes*/
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif

	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	if (rank == 0){
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}

	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
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
	th[0] = acos(X[3]/r[0]);
	phi[0] = atan2(X[2],X[1]);
}

/*Calculates covariant vector components after vector is rotated from (r, th, phi) to (pos_new[1], pos_new[2], pos_new[3]) over angle tilt*/
void rotate_vector(double V[NDIM],double pos_new[NDIM], double *r, double *th, double *phi, double tilt){
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
	double bl_gcov[NDIM][NDIM],bl_gcon[NDIM][NDIM],bl_gcon1[NDIM][NDIM], bl_gcon2[NDIM][NDIM], dxdxp[NDIM][NDIM],dxpdx[NDIM][NDIM], gdet1, gdet2;
	double V_tmp[NDIM],X[NDIM], X_tmp[NDIM], pos_new_tmp[NDIM];
	double theta_solve, theta_old, derivative;
	double delta_X2 = 0.1*M_PI / (double)N2*2. / M_PI;
	int step = 0;
	int i, j, k, l;
	for (i = 1; i < NDIM; i++){
		V_tmp[i] = V[i];
		pos_new_tmp[i] = pos_new[i];
	}

	/*Calculate length of vector wrt orthonormal basis instead of coordinate basis*/
	X[1] = pow(log(*r-RB), 1. / RADEXP);
	X[2] = 2. / M_PI*(*th) - 1.;
	X[3] = *phi;
	do{
		bl_coord(X, &(*r), &(theta_solve), &(*phi));
		theta_solve -= *th;
		theta_old = theta_solve;
		X[2] += delta_X2;
		bl_coord(X, &(*r), &(theta_solve), &(*phi));
		theta_solve -= *th;
		derivative = (theta_solve - theta_old) / delta_X2;
		X[2] -= theta_solve / derivative;
		step++;
	} while (fabs(theta_solve)>2.*M_PI / (double)N1 && step<3);
	kerr_gcov_func(*r, *th, bl_gcov);
	invert_matrix(bl_gcov, bl_gcon);
	dxdxp_func(X, dxdxp);
	invert_matrix(dxdxp, dxpdx);

	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			bl_gcon1[i][j] = 0;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					bl_gcon1[i][j] += bl_gcon[k][l] * dxpdx[i][k] * dxpdx[j][l];
				}
			}
		}
	}
	gdet1 = gdet_func(bl_gcon1);
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
	X[1] = pow(log(pos_new[1]-RB), 1. / RADEXP);
	X[2] = 2. / M_PI*pos_new[2] - 1.;
	X[3] = pos_new[3];
	step = 0;
	do{
		bl_coord(X, &(pos_new[1]), &(theta_solve), &(pos_new[3]));
		theta_solve -= pos_new[2];
		theta_old = theta_solve;
		X[2] += delta_X2;
		bl_coord(X, &(pos_new[1]), &(theta_solve), &(pos_new[3]));
		theta_solve -= pos_new[2];
		derivative = (theta_solve - theta_old) / delta_X2;
		X[2] -= theta_solve / derivative;
		step++;
	} while (fabs(theta_solve)>2.*M_PI / (double)N1 && step<3);
	kerr_gcov_func(pos_new[1], pos_new[2], bl_gcov);
	invert_matrix(bl_gcov, bl_gcon);

	dxdxp_func(X, dxdxp);
	invert_matrix(dxdxp, dxpdx);

	for (i = 0; i<NDIM; i++){
		for (j = 0; j<NDIM; j++){
			bl_gcon2[i][j] = 0;
			for (k = 0; k<NDIM; k++) {
				for (l = 0; l<NDIM; l++){
					bl_gcon2[i][j] += bl_gcon[k][l] * dxpdx[i][k] * dxpdx[j][l];
				}
			}
		}
	}
	//gdet2 = gdet_func(bl_gcov2);
	V[1] = (X_tmp[1] * sin(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * sin(pos_new[2])*sin(pos_new[3]) + X_tmp[3] * cos(pos_new[2]))/sqrt(bl_gcon2[1][1]);
	V[2] = (X_tmp[1] * cos(pos_new[2])*cos(pos_new[3]) + X_tmp[2] * cos(pos_new[2])*sin(pos_new[3]) - X_tmp[3] * sin(pos_new[2]))/sqrt(bl_gcon2[2][2]);
	V[3] = (-X_tmp[1] * sin(pos_new[3]) + X_tmp[2] * cos(pos_new[3]))/sqrt(bl_gcon2[3][3]);
}
void init_monopole(double Rout_val)
{
	printf("Error. Monopole not implemented in this version\n");
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

	AA = geom.gcov[TT][TT];
	BB = 2.*(geom.gcov[TT][1] * ucon[1] +
		geom.gcov[TT][2] * ucon[2] +
		geom.gcov[TT][3] * ucon[3]);
	CC = 1. +
		geom.gcov[1][1] * ucon[1] * ucon[1] +
		geom.gcov[2][2] * ucon[2] * ucon[2] +
		geom.gcov[3][3] * ucon[3] * ucon[3] +
		2.*(geom.gcov[1][2] * ucon[1] * ucon[2] +
		geom.gcov[1][3] * ucon[1] * ucon[3] +
		geom.gcov[2][3] * ucon[2] * ucon[3]);

	discr = BB*BB - 4.*AA*CC;
	ucon[TT] = (-BB - sqrt(discr)) / (2.*AA);
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
	//printf("(%d, %d, %d) Ratio 1: %f Ratio 2: %f Ratio 3: %f \n", ii, jj, zz, utconp[1], utconp[2] / old[2], utconp[3]/old[3]);
	/* done! */
}

void coord_transform2(double *V, int ii, int jj, int zz)
{

}

double lfish_calc(double r)
{
	return(
   ((pow(a,2) - 2.*a*sqrt(r) + pow(r,2))*
      ((-2.*a*r*(pow(a,2) - 2.*a*sqrt(r) + pow(r,2)))/
         sqrt(2.*a*sqrt(r) + (-3. + r)*r) +
        ((a + (-2. + r)*sqrt(r))*(pow(r,3) + pow(a,2)*(2. + r)))/
         sqrt(1 + (2.*a)/pow(r,1.5) - 3./r)))/
    (pow(r,3)*sqrt(2.*a*sqrt(r) + (-3. + r)*r)*(pow(a,2) + (-2. + r)*r))
	) ;
}
