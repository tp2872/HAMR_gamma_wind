#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"


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

#define DENSITY_NORMALIZATION NORMALIZE_BY_DENSITY_MAX

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


	#if (GPU_ENABLED)
	for (n = 0; n < n_active; n++) GPU_write(n_ord[n]);
	#endif
}


//note that only axisymmetric A is supported
void set_uniform_Bphi(void){
	int n, i, j, z;
	struct of_geom geom;
	#if(BOUND_TYPE2 == TRANSMISSIVE && STAGGERED)
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
	#if(BOUND_TYPE2 == TRANSMISSIVE && STAGGERED)
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
