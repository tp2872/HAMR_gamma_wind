/*
 *
 * generates initial conditions for a fishbone & moncrief disk 
 * with exterior at minimum values for density & internal energy.
 *
 * cfg 8-10-01
 *
 */
#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

void init()
{
	switch( WHICHPROBLEM ) {
		case MONOPOLE_PROBLEM_1D:
		case MONOPOLE_PROBLEM_2D:
			//init_monopole(1e3);
			break;
		case BZ_MONOPOLE_2D:
			//init_monopole(100.);
			break;
		case TORUS_PROBLEM:
			init_torus();
			break;
		case SPHERICAL_PROBLEM:
			init_torus_spherical();
			break;
		case THIN_PROBLEM:
			init_thindisk();
			break;
		case DISRUPTION_PROBLEM:
			init_disruption();
			break;
		case TORUS_PROBLEM_GRB:
			init_torus_grb();
			break;
		case POSTMERGER_PROBLEM:
			init_postmerger();
			break;
		case BONDI_PROBLEM_1D:
		case BONDI_PROBLEM_2D:
			init_bondi();
			break;
		case SOUND_WAVE:
			init_sndwave();
			break;
		case ENT_WAVE:
			init_entwave();
			break;
		case TRUNC_PROBLEM:
			//init_truncdisk();
		case BLAST_WAVE:
			init_blastwave();
			break;
		case SHOCK_TUBE:
			init_shocktube();
			break;
		case RAD_PULSE:
			init_radpulse();
			break;
		case COLLAPSAR:
			init_collapsar();
			break;
		case ISOLATED_NS:
			init_NS();
			break;
                case HLA_PULSAR:
                        init_wind_NS();
                        break;
                case CE_PULSAR:
                        init_wind_NS();
                        break;
	}

	int n;
	#if(GPU_ENABLED || GPU_DEBUG )
	for (n = 0; n < n_active;n++) GPU_write(n_ord[n]);
	GPU_boundprim(1);
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
	double tau, taumax, cell_size, kappa_abs, kappa_emmit, kappa_es;
	struct of_geom geom ;

	/* for disk interior */
	double l,rin,lnh,expm2chi,up1 ;
	double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
	double kappa,hm1, gam_local ;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;

	/* disk parameters (use fishbone.m to select new solutions) */
	double temp = a;
	a = 0.9375;
	rin = 10.0;
	rmax = 40.0;
	l = lfish_calc(rmax) ;
	kappa = 1.e-3 ;
	beta = BETA ;
	#if(RAD_M1)
	gam_local = 4. / 3.;
	#else
	gam_local = GAMMA;
	#endif

	//Check if there are enough cells within event horizon
	coord(0,5, 0, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);
	if (rank == 0) {
		fprintf(stderr, "r[5]: %g\n", r);
		fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - BH_SPIN*BH_SPIN)));
		if (r > 1. + sqrt(1. - BH_SPIN * BH_SPIN)) {
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
	taumax = 0.;
	#if(!NSY || CARTESIAN_GR)
	tilt = (TILT_ANGLE)/180.*M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif
	eccentricity = 0.0;
	double Tnu;
	for (n = 0; n < n_active; n++){
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z, tau, cell_size, kappa_abs, kappa_emmit, kappa_es, Tnu) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa, hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
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

			#if(RAD_M1)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1_RAD] = ur;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2_RAD] = uh;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3_RAD] = up;
			#endif

			/* regions outside torus */
			if(lnh < 0. || r < rin) {
				rho = 1.e-7*RHOMIN ;
				u = 1.e-7*UUMIN ;

				ur = 0.;
				uh = 0.;
				up = 0.;

				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][RHO] = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][UU] = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U3] = up;

				#if (DO_YE)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.5;
				#endif
				#if (DONUCLEAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 1.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 1.0;
				#endif			
			}
			/* region inside magnetized torus; u^i is calculated in
			 * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			 * so it needs to be transformed at the end */
			else { 
				hm1 = exp(lnh) - 1. ;
				rho = pow(hm1*(gam_local - 1.)/(kappa* gam_local),
							1./(gam_local - 1.)) ;
				u = kappa*pow(rho, gam_local)/(gam_local - 1.) ;
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
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u *(1. + 4.e-2 * (ranc(0) - 0.5));
				if(u > umax && r > rin){
					#pragma omp critical
					umax = u ;
				}
			
				#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
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

				#if (DO_YE)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.1;
				#endif
				#if (DONUCLEAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 0.0;
				#endif
			}

			// Setting temperature given u
			#if (DOHELM_TEMPERATURE == 2)
			eos_mode_rhou_temp_init(rho, &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], u);
			#endif

			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B3] = 0.;	

			// initialize neutrinos
			#if (NEUTRINOS_M1)
			for (int sp = 0; sp < NU_SPECIES; sp++) {
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] = 1e-15;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U1_NU, sp)] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U2_NU, sp)] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U3_NU, sp)] = up;

				Tnu = pow(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(NUMBER_NU, sp)] = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tnu);
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][NUMBER_NU] = 1e-30;
			}
			#endif
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

	//double torus_mass = 0.;
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][RHO] /= rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][UU] /= rhomax;

			#if(RAD_M1)
			//Set radiation pressure
			init_rad_pres(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
			#endif

			#if(TWO_T)
			double ue, ui, Theta, C;
			ue = 0.5 * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			if (ue < FLOOR_ENTROPY * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]) {
				ue = FLOOR_ENTROPY * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			ui = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] - ue;

			//Calculate electron entropy
			C = ue / p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO]);
				#else
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRE] = Theta * (Theta + 0.4) / pow(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], 2. / 3.);
				#endif

			//Calculate ion entropy
			C = ui / p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO]);
				#else
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRI] = Theta * (Theta + 0.4) / pow(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], 2. / 3.);
				#endif
			#endif
		}
	}

	//if (rank == 0) {
		//fprintf(stderr, "torus mass: %g\n", torus_mass);
	//}

	umax /= rhomax ;
	rhomax = 1. ;

	//Calculate maximum optical depth in once cell
	#if (MPI_enable)
	#if(RAD_M1)
	MPI_Allreduce(MPI_IN_PLACE, &taumax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#endif
	#endif

	#if(RAD_M1)
	//Print maximum optical depth in grid
	if (rank == 0) {
		fprintf(stderr, "taumax: %g\n", taumax);
	}
	#endif

	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}

	bound_prim(p, 1, t);

	set_mag();

	sourceflag=0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif

	#if (DOHELM)
	// Using density and pressure = (gam - 1) * u, find new u, using Helmholtz EOS
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] *= (gam_local - 1.);
			eos_mode_rhopres_u(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}

	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);
	
		#if (DOHELM_TEMPERATURE)
		// Set temperatures given u:
		for (n = 0; n < n_active; n++) {
			#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
				eos_mode_rhou_temp_init(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]
					#if (DONUCLEAR)
					, &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA], &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM]
					#endif
				);
			}
		}
		#endif
	#endif

	/* Initialize alpha particles */
	#if(DONUCLEAR)
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			init_nuclear(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}
	#endif

	/* Initialize neutrinos */
	#if(NEUTRINOS_M1)
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			//init_nuclear(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
			init_neutrinos(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}
	#endif

	//for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);

	/* initialize the entropies for two temperature fluids (electrons and ions) */
	#if(TWO_T)
	double bsq;

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
			set_2T_entropy(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], bsq);
		}
	}
	#endif

	//Set constant boundary conditions
	#if(CONSTANT_BC)
	int i1, k;
	for (n = 0; n < n_active; n++) {
		if (block[n_ord[n]][AMR_NBR2] == -1) {
			for (j = N2_GPU_offset[n_ord[n]]; j < N2_GPU_offset[n_ord[n]] + BS_2; j++)for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3; z++) {
				for (i1 = 0; i1 < N1G; i1++) {
					PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 + i1, j, z)][k] = p[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 - 1, j, z)][k];
					ps[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 + i1, j, z)][2] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 - 1, j, z)][2];
					ps[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 + i1, j, z)][3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1_GPU_offset[n_ord[n]] + BS_1 - 1, j, z)][3];
				}
			}
		}
	}
	#endif

	bound_prim(p, 1, t);

}

double lfish_calc(double r)
{
	return(
		((pow(a, 2) - 2. * a * sqrt(r) + pow(r, 2)) *
			((-2. * a * r * (pow(a, 2) - 2. * a * sqrt(r) + pow(r, 2))) /
				sqrt(2. * a * sqrt(r) + (-3. + r) * r) +
				((a + (-2. + r) * sqrt(r)) * (pow(r, 3) + pow(a, 2) * (2. + r))) /
				sqrt(1 + (2. * a) / pow(r, 1.5) - 3. / r))) /
		(pow(r, 3) * sqrt(2. * a * sqrt(r) + (-3. + r) * r) * (pow(a, 2) + (-2. + r) * r))
		);
}





