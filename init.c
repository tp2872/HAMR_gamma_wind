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

/*
 *
 * generates initial conditions for a fishbone & moncrief disk 
 * with exterior at minimum values for density & internal energy.
 *
 * cfg 8-10-01
 *
 */

#include "decs.h"


void coord_transform(double *pr, int ii, int jj, int zz);


void init()
{
  void init_bondi(void);
  void init_torus(void);
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
  case BONDI_PROBLEM_1D:
  case BONDI_PROBLEM_2D:
    init_bondi();
    break;
  }

}

void init_torus()
{
	int i,j,z ;
	double r,th,phi,sth,cth ;
	double ur,uh,up,u,rho ;
	double X[NDIM] ;
	struct of_geom geom ;
	/* for disk interior */
	double l,rin,lnh,expm2chi,up1 ;
	double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
	double kappa,hm1 ;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;
	double rmax, lfish_calc(double rmax) ;

	/* some physics parameters */
	gam = 5./3. ;

	/* disk parameters (use fishbone.m to select new solutions) */
    a = 0.9375 ;
    rin = 6. ;
	rmax = 12. ;
    l = lfish_calc(rmax) ;
	kappa = 1.e-3 ;
	beta = 100. ;

    /* some numerical parameters */
    lim = MC ;
    failed = 0 ;	/* start slow */
    cour = 0.9 ;
    dt = 1.e-5 ;
	R0 = 0.0 ;
    Rin = 0.8*(1. + sqrt(1. - a*a)) ;
    Rout = 100000. ;
    t = 0. ;
    hslope = 0.6 ;

	if(N2!=1) {
	  //2D problem, use full pi-wedge in theta
	  fractheta = 1.;
	}
	else{
	  //1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
	  fractheta = 1.e-2;
	}

    set_arrays() ;
    set_grid() ;

	coord(5, 0, 0, CENT, X);
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
	DTd = 25.0 ;	/* dumping frequency, in units of M */
	DTl = 25.0;	/* logfile frequency, in units of M */
	DTi = 25.0; 	/* image file frequ., in units of M */
	DTr = 5.0 * 1000.; 	/* restart file frequ., in timesteps */

	/* start diagnostic counters */
	dump_cnt = 0 ;
	image_cnt = 0 ;
	rdump_cnt = 0 ;
	defcon = 1. ;

	rhomax = 0. ;
	umax = 0. ;
	ZSLOOP3D(N1_MPI_offset, N1_MPI + N1_MPI_offset - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) {
		coord(i, j, z, CENT, X);
		bl_coord(X,&r,&th, &phi) ;

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

			/* these values are demonstrably physical
			   for all values of a and r */
			/*
                        ur = -1./(r*r) ;
                        uh = 0. ;
			up = 0. ;
			*/

			ur = 0. ;
			uh = 0. ;
			up = 0. ;

			/*
			get_geometry(i,j,CENT,&geom) ;
                        ur = geom.gcon[0][1]/geom.gcon[0][0] ;
                        uh = geom.gcon[0][2]/geom.gcon[0][0] ;
                        up = geom.gcon[0][3]/geom.gcon[0][0] ;
			*/

			p[index(i,j,z)][RHO] = rho;
			p[index(i,j,z)][UU] = u;
			p[index(i,j,z)][U1] = ur;
			p[index(i,j,z)][U2] = uh;
			p[index(i,j,z)][U3] = up;
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

			p[index(i,j,z)][RHO] = rho;
			if(rho > rhomax) rhomax = rho ;
			p[index(i,j,z)][UU] = u*(1. + 4.e-2*(ranc(0) - 0.5));
			if(u > umax && r > rin) umax = u ;
			p[index(i,j,z)][U1] = ur;
			p[index(i,j,z)][U2] = uh;
			p[index(i,j,z)][U3] = up;
			
			/* convert from 4-vel to 3-vel */
			coord_transform(p[index(i,j,z)], i, j,z);
			
		}
		p[index(i,j,z)][B1] = 0.;
		p[index(i,j,z)][B2] = 0.;
		p[index(i, j, z)][B3] = 0.;
	}
	
	/*Share rhomax among MPI processes*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&rhomax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Irecv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs[i]);
			MPI_Wait(&reqs[i], Stat);
			rhomax = MY_MAX(rhomax, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&rhomax, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}

	if (rank != 0){
		rc = MPI_Irecv(&rhomax, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank + 2 * numtasks]);
		MPI_Wait(&reqs[rank + 2 * numtasks], Stat);
	}
	
	/*Share umax among MPI processes*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&umax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Irecv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs[i]);
			MPI_Wait(&reqs[i], Stat);
			umax = MY_MAX(umax, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&umax, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}

	if (rank != 0){
		rc = MPI_Irecv(&umax, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank + 2 * numtasks]);
		MPI_Wait(&reqs[rank + 2 * numtasks], Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0){
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	//ZSLOOP(0,N1-1,0,N2-1) {
	ZSLOOP3D(N1_MPI_offset, N1_MPI + N1_MPI_offset - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) {
		p[index(i,j,z)][RHO] /= rhomax ;
		p[index(i,j,z)][UU]  /= rhomax ;
	}
	umax /= rhomax ;
	rhomax = 1. ;
	fixup(p);
	bound_prim(p,1);
	bound_prim(p,1);

	/* first find corner-centered vector potential */
	ZSLOOP3D(N1_MPI_offset, N1_MPI + N1_MPI_offset, N2_MPI_offset, N2_MPI_offset + N2_MPI, N3_MPI_offset, N3_MPI_offset + N3_MPI){
		dq[index(i,j,z)][0] = 0.;
	}
	ZSLOOP3D(N1_MPI_offset, N1_MPI + N1_MPI_offset, N2_MPI_offset, N2_MPI_offset + N2_MPI, N3_MPI_offset, N3_MPI_offset + N3_MPI){
                /* vertical field version */
                /*
                coord(i,j,CORN,X) ;
                bl_coord(X,&r,&th) ;

                A[i][j] = 0.5*r*sin(th) ;
                */

                /* field-in-disk version */
		/* flux_ct */
        rho_av = 0.25*(
             p[index(i,j,z)][RHO] +
			 p[index(i-1, j, z)][RHO] +
			 p[index(i, j-1, z)][RHO] +
			 p[index(i - 1, j-1, z)][RHO]);

        q = rho_av/rhomax - 0.2 ;
		if (q > 0.) dq[index(i, j, z)][0] = q;
    }


		/* now differentiate to find cell-centered B,
		and begin normalization */
	bsq_max = 0.;
	ZLOOP3D_MPI{
		get_geometry(i, j, CENT, &geom);

		/* flux-ct */
		p[index(i, j, z)][B1] = -(dq[index(i, j, z)][0] - dq[index(i, j+1, z)][0]
			+ dq[index(i+1, j, z)][0] - dq[index(i+1, j+1, z)][0]) / (2.*dx[2] * geom.g);
		p[index(i, j, z)][B2] = (dq[index(i, j, z)][0] + dq[index(i, j+1, z)][0]
			- dq[index(i+1, j, z)][0] - dq[index(i+1, j+1, z)][0]) / (2.*dx[1] * geom.g);

		p[index(i,j,z)][B3] = 0.;

		bsq_ij = bsq_calc(p[index(i,j,z)], &geom);
		if (bsq_ij > bsq_max) bsq_max = bsq_ij;
	}

	/*Share bsq_max among MPI processes*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&bsq_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Irecv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs[i]);
			MPI_Wait(&reqs[i], Stat);
			bsq_max = MY_MAX(bsq_max, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&bsq_max, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}
	if (rank != 0){
		rc = MPI_Irecv(&bsq_max, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank + 2 * numtasks]);
		MPI_Wait(&reqs[rank + 2 * numtasks], Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0){
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}

	/* finally, normalize to set field strength */
	beta_act = (gam - 1.)*umax/(0.5*bsq_max) ;
	if (rank == 0){
		fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	}
	norm = sqrt(beta_act/beta) ;
	bsq_max = 0. ;
	ZLOOP3D_MPI{
		p[index(i,j,z)][B1] *= norm;
		p[index(i,j,z)][B2] *= norm;

		get_geometry(i, j, CENT, &geom);
		bsq_ij = bsq_calc(p[index(i,j,z)], &geom);
		if (bsq_ij > bsq_max) bsq_max = bsq_ij;
	}


	/*Share bsq_max among MPI processes*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&bsq_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Irecv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs[i]);
			MPI_Wait(&reqs[i], Stat);
			bsq_max = MY_MAX(bsq_max, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&bsq_max, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}
	if (rank != 0){
		rc = MPI_Irecv(&bsq_max, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank + 2 * numtasks]);
		MPI_Wait(&reqs[rank + 2 * numtasks], Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	beta_act = (gam - 1.)*umax/(0.5*bsq_max) ;
	if(rank == 0){
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}

	/* enforce boundary conditions */
	fixup(p) ;
	bound_prim(p,1) ;
	bound_prim(p,1);

	#if( DO_FONT_FIX ) 
	set_Katm();
	#endif 
}

void init_bondi()
{
	printf("Error. Bondi not implemented in this version\n");
}

void init_monopole(double Rout_val)
{
	printf("Error. Monopole not implemented in this version\n");
}

/* this version starts w/ BL 4-velocity and
* converts to 3-velocities in modified
* Kerr-Schild coordinates */
void coord_transform(double *pr, int ii, int jj, int zz)
{
	double X[NDIM], r, th, phi, ucon[NDIM], trans[NDIM][NDIM], tmp[NDIM];
	double AA, BB, CC, discr;
	double alpha, gamma, beta[NDIM];
	struct of_geom geom;
	struct of_state q;
	int i, j, k, m;

	coord(ii, jj, zz, CENT, X);
	bl_coord(X, &r, &th, &phi);
	blgset(ii, jj, &geom);

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
	ucon[1] *= (1. / (r - R0));
	ucon[2] *= (1. / (M_PI + (1. - hslope)*M_PI*cos(2.*M_PI*X[2])));
	//ucon[3] *= 1.; //!!!ATCH: no need to transform since will use phi = X[3]

	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	get_geometry(ii, jj, CENT, &geom);
	alpha = 1. / sqrt(-geom.gcon[0][0]);
	gamma = ucon[TT] * alpha;

	beta[1] = alpha*alpha*geom.gcon[0][1];
	beta[2] = alpha*alpha*geom.gcon[0][2];
	beta[3] = alpha*alpha*geom.gcon[0][3];

	pr[U1] = ucon[1] + beta[1] * gamma / alpha;
	pr[U2] = ucon[2] + beta[2] * gamma / alpha;
	pr[U3] = ucon[3] + beta[3] * gamma / alpha;

	/* done! */
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


