#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"


void init_NS()
{
	int i, j, z, n;
	double r, th, phi, sth, cth, rhor;
	double ur, uh, up, u, rho;
	double bl_gcov[NDIM][NDIM];
	double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
	double tilt, eccentricity;
	double tau, taumax, cell_size, kappa_abs, kappa_emmit, kappa_es;
	struct of_geom geom;
	if (rank == 0) {
		fprintf(stderr, "start init NS\n");
	}
	/* for disk interior */
	double l, rin, lnh, expm2chi, up1;
	double DD, AA, SS, thin, sthin, cthin, DDin, AAin, SSin;
	double kappa, hm1, gam_local;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av, rhomax, umax, beta, bsq_ij, bsq_max, norm, q, beta_act;

	/* disk parameters (use fishbone.m to select new solutions) */

#if(RAD_M1)
	gam_local = 4. / 3.;
#else
	gam_local = 4. / 3.;
#endif

	//Check if there are enough cells within event horizon
	coord(0, 5, 0, 0, CENT, X);
	bl_coord(X, &r, &th, &phi);
	if (rank == 0) {
		fprintf(stderr, "r[5]: %g\n", r);
#if(NEUTRON_STAR)
		fprintf(stderr, "r[5]/RNS: %g \n", r / R_NS);
#else
		fprintf(stderr, "r[5]/rhor: %g", r / (1. + sqrt(1. - BH_SPIN * BH_SPIN)));

		if (r > (1. + sqrt(1. - BH_SPIN * BH_SPIN))) {
			fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n");
		}
		else {
			fprintf(stderr, "\n");
		}
#endif
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
	taumax = 0.;
#if(!NSY || CARTESIAN_GR)
	tilt = (TILT_ANGLE) / 180. * M_PI;
#else
	tilt = -(TILT_ANGLE) / 180. * M_PI;
#endif
	eccentricity = 0.0;
	double Tnu;
	if (rank == 0) {
		fprintf(stderr, "start NS ICs\n");
	}
	for (n = 0; n < n_active; n++) {
#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z, tau, cell_size, kappa_abs, kappa_emmit, kappa_es, Tnu) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa, hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;
#if (TILTED && 0)
			sph_to_cart(X_cart, &(pos_new[1]), &(pos_new[2]), &(pos_new[3]));
			rotate_coord(X_cart, -tilt);
			cart_to_sph(X_cart, &r, &th, &phi);
#endif

			rho = 1.e-20 * RHOMIN;
			u = 1.e-20 * UUMIN;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u;
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = -sqrt(-1.0 / geom.gcov[0][0]) * geom.gcon[0][1] / geom.gcon[0][0]; // Make static wrt coordinates
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = -sqrt(-1.0 / geom.gcov[0][0]) * geom.gcon[0][2] / geom.gcon[0][0];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = -sqrt(-1.0 / geom.gcov[0][0]) * geom.gcon[0][3] / geom.gcon[0][0];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLR] = 1.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLRFRAC] = 1.0;

			if (rho > rhomax) {
#pragma omp critical
				rhomax = rho;
			}
			if (u > umax) {
#pragma omp critical
				umax = u;
			}
#if (DO_YE)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.5;
#endif
#if (DONUCLEAR)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 1.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 1.0;
#endif	
			// Setting temperature given u
#if (DOHELM_TEMPERATURE == 2)
			eos_mode_rhou_temp_init(rho, &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], u);
#endif

			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][PS1START] = 0.;
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
	
#if (MPI_enable)
	/*Share rhomax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	/*Share umax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#endif

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0) {
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
#if((NEUTRON_STAR))
	//double torus_mass = 0.;
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] /= rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] /= rhomax;

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
#endif
	//if (rank == 0) {
		//fprintf(stderr, "torus mass: %g\n", torus_mass);
	//}

	umax /= rhomax;
	rhomax = 1.;

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

	//for (n = 0; n < n_active; n++) {
	//	fixup(p, n_ord[n]);
	//}

	//bound_prim(p, 1);
	if (rank == 0) {
		fprintf(stderr, "start set mag NS\n");
	}
	set_mag_NS();

#if(NS_TORUS)
	if (rank == 0) {
		fprintf(stderr, "start set torus ICs\n");
	}

	init_NS_torus();
	int k=0;
	for (n = 0; n < n_active; n++) {
#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z,k)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k] += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];
		}
	}

	if (rank == 0) {
		fprintf(stderr, "start set torus mag\n");
	}
	set_mag_torus();

#endif

	sourceflag = 0.;
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
	//for (n = 0; n < n_active; n++) {
	//	fixup(p, n_ord[n]);
	//}
	//fprintf(stderr, "after boundprim \n");
#if(GPU_ENABLED && NEUTRON_STAR)
	for (n = 0; n < n_active; n++) {
		/*Radial magentic field at the face center of surface cell; nope save the initial face center B field*/
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, N1G - 1 + BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + N2G -1, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
			//coord(n_ord[n], i, j, z, FACE1, X);
			//bl_coord(X, &r, &th, &phi);
			//get_geometry(n_ord[n], i, j, z, FACE1, &geom);
#if(USE_PS1START && STAGGERED)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][PS1START] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][PS1START] = geom.g * ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
#else
			double dxdxp_FAZ[NDIM][NDIM], dxdxp_surf[NDIM][NDIM], dxpdx_surf[NDIM][NDIM];
			double r_surf, r_FAZ;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			r_FAZ = r;
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			dxdxp_func(X, dxdxp_FAZ);
			coord(n_ord[n], i, j, z, FACE1, X);
			bl_coord(X, &r, &th, &phi);
			r_surf = r;
			get_geometry(n_ord[n], i, j, z, FACE1, &geom);
			dxdxp_func(X, dxdxp_surf);                                         
			invert_matrix(dxdxp_surf, dxpdx_surf);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][PS1START] = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] * pow(r_FAZ / r_surf, 4) * dxpdx_surf[1][1] * dxdxp_FAZ[1][1];
			//Bx1_surface[nl[n]][index_3D(n, i, j, z)] = geom.g * ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			//Bx1_surface_GPU[nl[n_ord[n]]][(j - N2_GPU_offset[n_ord[n]] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n_ord[n]] + N3G)] = geom.gdet * ps[nl[n_ord[n]]][index_3D(n_ord[n], 0, j, z)][1];
#endif
			
		}
	}
#endif
}

void set_mag_NS(void) {
	int i, j, z, k, n;
	double rhomax = 1., pmax = 0.;
	int i100 = 0;
	double  beta = (GAMMA - 1.0) / BSQOUMAX;
	double rho_av, q, bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM], pos_new[NDIM], beta_ij;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	struct of_state state;
	double gamma_g;

#if(!NSY || CARTESIAN_GR)
	double tilt = (TILT_ANGLE) / 180. * M_PI;
#else
	double tilt = -(TILT_ANGLE) / 180. * M_PI;
#endif	


	do {
		i100++;
		coord(0, i100, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
	} while (r < ROUT);

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + D3) {
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
#if(STAGGERED)
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
#endif
		}
	}

	/* first find corner-centered vector potential */
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + D3) {
			/* Cell centered vector potential */
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
#if (TILTED && 0)		
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;
			sph_to_cart(X_cart, &r, &th, &phi);
			rotate_coord(X_cart, -tilt);
			cart_to_sph(X_cart, &r, &th, &phi);
#endif
#if(NEUTRON_STAR)
			double z1, z1inv, schwFactor, A_schw;
			z1 = 2.0 / r;
			z1inv = 1.0 / z1;
			schwFactor = 0.5 + z1inv + z1inv * z1inv * log(1.0 - z1);
			A_schw = -schwFactor * 3.0 * MU_NS / 2.0 * pow(R_NS / 4.0, 3.0);
#if(SPHERICAL_GR && OBLIQUE_NS)
			double sin_chi, cos_chi;
#if(DEFORM_DIPOLE_NS)
			/*** Makes an aligned dipole beyond some radius r1,  ***
			 *** and a misaligned one inside another radius r0.  ***/
			double f;
			f = f_misalignment(r);
			sin_chi = sin(f * OBL_ANGLE_NS);
			cos_chi = cos(f * OBL_ANGLE_NS);
#else
			sin_chi = sin(OBL_ANGLE_NS);
			cos_chi = cos(OBL_ANGLE_NS);
#endif /* DEFORM_DIPOLE_NS */
			/* Change of basis: magnetic --> grid/rotational */
			double Atheta_ang = -sin_chi * sin(phi);
			double Aphi_ang = sin(th) * (cos_chi * sin(th) - sin_chi * cos(th) * cos(phi));

			/***
			if ( i==0 && r < 4.5 && k == 8 )
			  fprintf(stderr, "j: %d  theta: %.5e  Atheta: %.5e  Aphi: %.5e \n", j,
						  theta, Atheta_ang, Aphi_ang) ;
			 ***/

			 /* transform to code coords                                         */
			 /* dr^\mu/dx^\nu jacobian, where x^\nu are internal coords          */
			double dxdxp[NDIM][NDIM];
			dxdxp_func(X, dxdxp);

			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.0;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = (dxdxp[1][2] * Atheta_ang + dxdxp[1][3] * Aphi_ang) * A_schw;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = (dxdxp[2][2] * Atheta_ang + dxdxp[2][3] * Aphi_ang) * A_schw;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = (dxdxp[3][2] * Atheta_ang + dxdxp[3][3] * Aphi_ang) * A_schw;
#else
#if(SPHERICAL_GR)
			q = A_schw * sin(th) * sin(th);
#else
#if(MONOPOLE)
			q = MU_NS * (1.0 - cos(th)); // Monopole
#else
			q = MU_NS * sin(th) * sin(th) / r; // Flat spacetime dipole
#endif
#endif
			

			if (q > 0.) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = q; //SANE+CODE_COMPARISON
				
				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = pow(q, 2.0) * pow(r, 3.0); //MAD
			}
			else {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
			}
#endif
#endif

#if (TILTED && 0)
			V[1] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			V[2] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2];
			V[3] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			rotate_vector2(V, pos_new, &r, &th, &phi, tilt);
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = V[1];
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = V[2];
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = V[3];
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
			}
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
			}
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
			}
#endif

#if((SPHERICAL || SPHERICAL_GR) && !(OBLIQUE_NS) && !(MONOPOLE))
			if (j < 0 || j >= N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= -1.0;
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= -1.0;
			}
#endif
			//if (block[n_ord[n]][AMR_NBR4] == -1 && i == 0 && (j < 3 || j > N2 - 4) && z == 0) {
			//	fprintf(stderr, "B dq[1]: %g dq[2]: %g dq[3]: %g for r=%g %d, th=%g %d, phi=%g %d\n", dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2], dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3], r, i, th, j, phi, z);
			//}
#if(!(OBLIQUE_NS && NEUTRON_STAR) && !(MONOPOLE))  //not sure about this
			double dxdxp[NDIM][NDIM], dq_temp[NDIM];
			int k1, k2;
			dxdxp_func(X, dxdxp);

			for (k = 0; k < NDIM; k++) dq_temp[k] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];

			for (k1 = 0; k1 < NDIM; k1++) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k1] = 0;
				for (k2 = 0; k2 < NDIM; k2++) {
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k1] += dxdxp[k2][k1] * dq_temp[k2];
				}
			}
#endif
			//if (block[n_ord[n]][AMR_NBR4] == -1 && i == 0 && (j < 3 || j > N2 - 4) && z == 0) {
			//	fprintf(stderr, "A dq[1]: %g dq[2]: %g dq[3]: %g for r=%g %d, th=%g %d, phi=%g %d\n", dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2], dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3], r, i, th, j, phi, z);
			//}
		}
	}

	//Transform from cell centered vector potential to edge centered vector potential
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - D1, BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] - D2, N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]] - D3, N3_GPU_offset[n_ord[n]] + BS_3) {
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z - D3)][1]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z - D3)][2]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j - D2, z)][3]);
			//if (block[n_ord[n]][AMR_NBR4] == -1 && i == 0 && (j < 3 || j > N2 - 4) && z == 0) {
			//	coord(n_ord[n], i, j, z, FACE1, X);
			//	bl_coord(X, &r, &th, &phi);
			//	fprintf(stderr, "B Ecorn[1]: %g Ecorn[2]: %g Ecorn[3]: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2], E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3], r, i, th, j, phi, z);
			//}
		}
	}

	/* now differentiate to find cell-centered B,
	and begin normalization */
#if(STAGGERED)
	gpu = 0;
	nstep = 2 * AMR_SWITCHTIMELEVEL - 1;
	set_prestep();
	const_transport_bound();
	nstep = 0;
#endif
	double th1, th2;
	for (n = 0; n < n_active; n++) {
#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
#if(NEUTRON_STAR && !(MONOPOLE))
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				if (z == 0 && i == 0) {
					for (int k2 = -N2G; k2 < N2G; k2++) {
						coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + k2, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						//fprintf(stderr, "B Ecorn[3] FACE2: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + k2, z)][3], r, i, th, N2_GPU_offset[n_ord[n]] + k2, phi, z);
					}
				}
				#if(NEUTRON_STAR && 0)
				int k2 = 0;
				double th1 = 0.0, th2 = 0.0;
				double dxdxp1[NDIM][NDIM], dxdxp2[NDIM][NDIM], E_corn_temp[NDIM];
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]], z, FACE1, X);
				bl_coord(X, &r, &th1, &phi);
				dxdxp_func(X, dxdxp1);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] +1, z, FACE1, X);
				bl_coord(X, &r, &th2, &phi);
				dxdxp_func(X, dxdxp2);
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0;
				for (k2 = 0; k2 < NDIM; k2++) {
					E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] += (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z)][k2] * dxdxp2[k2][3] * sin(th1) * sin(th1) / (dxdxp1[k2][3] * sin(th2) * sin(th2)));
				}
				//E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z)][3] * sin(th1) * sin(th1) / (sin(th2) * sin(th2));
				#elif(0)
				
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]], z, FACE2, X);
				bl_coord(X, &r, &th1, &phi);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z, FACE2, X);
				bl_coord(X, &r, &th2, &phi);
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z)][3] * sin(th1) * sin(th1) / (sin(th2) * sin(th2));
				for (int k2 = 0; k2 < N2G; k2++) {
					E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] - 1 - k2, z)][3] = -E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + k2 + 1, z)][3];
				}
							
				#else
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
				#endif
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = 0.;
				if (z == 0 && i == 0) {
					for (int k2 = -N2G; k2 < N2G; k2++) {
						coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + k2, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						//fprintf(stderr, "A Ecorn[3] FACE2: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + k2, z)][3], r, i, th, N2_GPU_offset[n_ord[n]] + k2, phi, z);
					}
				}
			}
		}

		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				if (z == 0 && i == 0) {
					for (int k2 = -N2G; k2 < N2G; k2++) {
						coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - k2, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						//fprintf(stderr, "B Ecorn[3] FACE2: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - k2, z)][3], r, i, th, N2_GPU_offset[n_ord[n]] + BS_2 - k2, phi, z);
					}
				}
				#if(NEUTRON_STAR && 0)
				int k1 = 0, k2 = 0;
				double th1 = 0.0, th2 = 0.0;
				double dxdxp1[NDIM][NDIM], dxdxp2[NDIM][NDIM], dxpdx2[NDIM][NDIM], E_corn_temp1[NDIM], E_corn_temp2[NDIM];
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z, CORN, X);
				bl_coord(X, &r, &th1, &phi);
				dxdxp_func(X, dxdxp1);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z, CORN, X);
				bl_coord(X, &r, &th2, &phi);
				dxdxp_func(X, dxdxp2);
				invert_matrix(dxdxp2, dxpdx2);
				for (k1 = 0; k1 < NDIM; k1++) {
					E_corn_temp2[k1] = 0.0;
					for (k2 = 0; k2 < NDIM; k2++) {
						E_corn_temp2[k1] += (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][k2] * dxpdx2[k2][k1]);
					}
				}
				for (k1 = 0; k1 < NDIM; k1++) {
					E_corn_temp1[k1] = E_corn_temp2[k1] * sin(th1) * sin(th1) / (sin(th2) * sin(th2));
				}
				for (k1 = 0; k1 < NDIM; k1++) {
					E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][k1] = 0.0;
					for (k2 = 0; k2 < NDIM; k2++) {
						E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][k1] += (E_corn_temp1[k2] * dxdxp1[k2][k1]);
					}
				}
				if (i == 0 && (j < N2_GPU_offset[n_ord[n]]+3 || j > N2_GPU_offset[n_ord[n]] + BS_2 - 4) && z == 0) {
					get_geometry(n_ord[n], i, j-1, z, FACE1, &geom);
					fprintf(stderr, "E_corn_temp1[1]: %g E_corn_temp1[2]: %g E_corn_temp1[3]: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn_temp1[1], E_corn_temp1[2], E_corn_temp1[3], r, i, th1, j, phi, k);
					fprintf(stderr, "E_corn_temp2[1]: %g E_corn_temp2[2]: %g E_corn_temp2[3]: %g for r=%g %d, th=%g %d, phi=%g, dx2=%g\n", E_corn_temp2[1], E_corn_temp2[2], E_corn_temp2[3], r, i, th2, j, phi, k, dx[nl[n_ord[n]]][2]);
				}
				//E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][3] * sin(th1) * sin(th1) / (sin(th2) * sin(th2)));
				
				
				//E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][3] * sin(th1) * sin(th1) / (sin(th2) * sin(th2));
				#elif(0)
				
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z, FACE2, X);
				bl_coord(X, &r, &th1, &phi);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z, FACE2, X);
				bl_coord(X, &r, &th2, &phi);
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][3] * sin(th1) * sin(th1) / (sin(th2) * sin(th2));
				for (int k2 = 0; k2 < N2G; k2++) {
					E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 + k2 + 1, z)][3] = -E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - k2 - 1, z)][3];
				}
				
				
				#else
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.;
				#endif
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][1] = 0.;

				if (z == 0 && i == 0) {
					for (int k2 = -N2G; k2 < N2G; k2++) {
						coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - k2, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						//fprintf(stderr, "A Ecorn[3] FACE2: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - k2, z)][3], r, i, th, N2_GPU_offset[n_ord[n]] + BS_2 - k2, phi, z);
					}
				}
			}
		}
#endif
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			//if (block[n_ord[n]][AMR_NBR4] == -1 && i == 0 && (j < 3 || j > N2 - 4) && z == 0) {
			//	coord(n_ord[n], i, j, z, FACE1, X);
			//	bl_coord(X, &r, &th, &phi);
			//	fprintf(stderr, "A Ecorn[1]: %g Ecorn[2]: %g Ecorn[3]: %g for r=%g %d, th=%g %d, phi=%g %d\n", E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2], E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3], r, i, th, j, phi, z);
			//}
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
#if(NEUTRON_STAR && 0)
		double th1, th2, r1, phi1;
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]], z, FACE1, X);
				bl_coord(X, &r1, &th1, &phi1);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z, FACE1, X);
				bl_coord(X, &r, &th2, &phi);
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z)][1] * cos(th1) / cos(th2);
				//if (i < 5 && z < 5) {
				//	fprintf(stderr, "ps change AMR_NBR1: ps1_th1=%g, ps1_th2=%g, r=%g(%d) th1=%g(%d) th2=%g(%d) phi=%g(%d)\n", ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1], ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + 1, z)][1], r1, i, th1, N2_GPU_offset[n_ord[n]], th2, N2_GPU_offset[n_ord[n]] + 1, phi1, z);
				//}
				
			}
		}
		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z, FACE1, X);
				bl_coord(X, &r1, &th1, &phi1);
				coord(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 2, z, FACE1, X);
				bl_coord(X, &r, &th2, &phi);
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][1] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 2, z)][1] * cos(th1) / cos(th2);
				//if (i < 5 && z < 5) {
				//	fprintf(stderr, "ps change AMR_NBR3: ps1_th1=%g, ps1_th2=%g, r=%g(%d) th1=%g(%d) th2=%g(%d) phi=%g(%d)\n", ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 1, z)][1], ps[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2 - 2, z)][1], r1, i, th1, N2_GPU_offset[n_ord[n]] + BS_2 - 1, th2, N2_GPU_offset[n_ord[n]] + BS_2 - 2, phi1, z);
				//}
			}
		}

#endif
	}

	double bsq_max = 0.;
	double ug_sum = 0.;
	double bsq_sum = 0.;
	for (n = 0; n < n_active; n++) {
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B1] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2. * dx[nl[n_ord[n]]][2] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2. * dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B2] =
				(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2. * dx[nl[n_ord[n]]][1] * geom.g)
				- (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2. * dx[nl[n_ord[n]]][3] * geom.g);
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B3] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2. * dx[nl[n_ord[n]]][1] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2. * dx[nl[n_ord[n]]][2] * geom.g);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE1] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i + D1, j, z)][FACE1]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE2] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j + D2, z)][FACE2]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#if(N3G>0)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = (ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE3] + ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + D3)][FACE3]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			#endif
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#endif
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);
			//fprintf(stderr, "initial B1 B2 B3 bsq_ij: %g %g %g %g\n", p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3], bsq_ij);
#if(NEUTRON_STAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = bsq_ij / pow(10.0, SURF_MAX_BSQ_RHO_LOG);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 0.2 * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO];
				beta_ij = 2.0 * (gam - 1.0) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] / bsq_ij;
#endif
			
			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0 / 3.0;
				#else
				gamma_g = GAMMA;
				#endif			
			#else
			gamma_g = GAMMA;
			#endif


			#if(RAD_M1)
			if (((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 4) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else

			if ((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && j > (int)(20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && j < (int)(N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				bsq_max = bsq_ij;
			}
			if (block[n_ord[n]][AMR_NBR4] == -1 && i == 0 && (j < 3 || j > N2 - 4) && z == 0) {
				coord(n_ord[n], i, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				//fprintf(stderr, "B: AMR_NBR4: ps[1]: %g for r=%g %d, th=%g %d, phi=%g %d\n", ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1],r,i,th,j,phi,z);
			}
			//if (block[n_ord[n]][AMR_NBR4] == -1 && i <5 && j==0 && z == 0) {
			//	coord(n_ord[n], i, j, z, CENT, X);
			//	bl_coord(X, &r, &th, &phi);
			//	fprintf(stderr, "AMR_NBR: ps[1]: %g for r=%g %d, th=%g %d, phi=%g %d\n", ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], r, i, th, j, phi, z);
			//}
			//if ((block[n_ord[n]][AMR_POLE] == 1) && i < 5 && z == 0) {
			//	coord(n_ord[n], i, j, z, CENT, X);
			//	bl_coord(X, &r, &th, &phi);
			//	fprintf(stderr, "AMR_NBR1: ps[1]: %g for r=%g %d, th=%g %d, phi=%g %d\n", ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1], r, i, th, j, phi, z);
			//}
		}
	}

#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#endif

	if (rank == 0) {
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}
	/*NEUTRON STAR INIT*/

	/* finally, normalize to set field strength */

	beta_act = pmax / (0.5 * bsq_max);
	if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++) {
		//ZLOOP3D_MPI{
		//	fprintf(stderr, "rho: %g\n", p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][RHO]);
		//}
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1, t);


#if(NEUTRON_STAR)
	norm = 1.0;// / bsq_max);
#else
	norm = sqrt(beta_act / beta);
#endif
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= norm;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= norm;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= norm;
#if(NEUTRON_STAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] *= (norm * norm);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 0.2 * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO];
#endif
#if(STAGGERED)
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] *= norm;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= norm;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= norm;
#endif	
		}
	}


#if(RESISTIVE)
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + D3) {
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			set_E_init(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], geom);
		}
	}
#endif

	bsq_max = 0.;
	pmax = 0;
	bsq_sum = 0.;
	ug_sum = 0.;
	for (n = 0; n < n_active; n++) {
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);

			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				bsq_max = bsq_ij;
			}
			//fprintf(stderr, "initial rho uu bsq_ij: %g %g %g\n", p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], bsq_ij);
			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0 / 3.0;
				#else
				gamma_g = GAMMA;
				#endif
			#else
			gamma_g = GAMMA;
			#endif

			#if(RAD_M1)
			if (((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else
			if ((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
		}
	}

	/*Share bsq_max among MPI processes*/
#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#endif

	beta_act = pmax / (0.5 * bsq_max);
	if (rank == 0) {
		fprintf(stderr, "final bsq_max: %g\n", bsq_max);
	}
	if (rank == 0) {
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}


}

#if(NEUTRON_STAR && NS_TORUS)
void init_NS_torus()
{
	int i, j, z, n;
	double r, th, phi, sth, cth;
	double ur, uh, up, u, rho;
	double bl_gcov[NDIM][NDIM];
	double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
	double tilt, eccentricity;
	double tau, taumax, cell_size, kappa_abs, kappa_emmit, kappa_es;
	struct of_geom geom;

	/* for disk interior */
	double l, rin, lnh, expm2chi, up1;
	double DD, AA, SS, thin, sthin, cthin, DDin, AAin, SSin;
	double kappa, hm1, gam_local;

	/*For MPI*/
	double inmsg;

	/* for magnetic field */
	double rho_av, rhomax, umax, beta, bsq_ij, bsq_max, norm, q, beta_act;

	/* disk parameters (use fishbone.m to select new solutions) */
	double temp = a;
	a = 0.9375;
	rin = 40.0;
	rmax = 60.0;
	l = lfish_calc(rmax);
	kappa = 1.e-3;
	beta = BETA;
#if(RAD_M1)
	gam_local = 4. / 3.;
#else
	gam_local = GAMMA;
#endif


	rhomax = 0.;
	umax = 0.;
	taumax = 0.;
#if(!NSY || CARTESIAN_GR)
	tilt = (TILT_ANGLE) / 180. * M_PI;
#else
	tilt = -(TILT_ANGLE) / 180. * M_PI;
#endif
	eccentricity = 0.0;
	double Tnu;
	for (n = 0; n < n_active; n++) {
#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z, tau, cell_size, kappa_abs, kappa_emmit, kappa_es, Tnu) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa, hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
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

			/* calculate lnh */
			DD = r * r - 2. * r + a * a;
			AA = (r * r + a * a) * (r * r + a * a) - DD * a * a * sth * sth;
			SS = r * r + a * a * cth * cth;

			thin = M_PI / 2.;
			sthin = sin(thin);
			cthin = cos(thin);
			DDin = rin * rin - 2. * rin + a * a;
			AAin = (rin * rin + a * a) * (rin * rin + a * a)
				- DDin * a * a * sthin * sthin;
			SSin = rin * rin + a * a * cthin * cthin;

			if (r >= rin) {
				lnh = 0.5 * log((1. + sqrt(1. + 4. * (l * l * SS * SS) * DD /
					(AA * sth * AA * sth))) / (SS * DD / AA))
					- 0.5 * sqrt(1. + 4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth))
					- 2. * a * r * l / AA
					- (0.5 * log((1. + sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
						(AAin * AAin * sthin * sthin))) / (SSin * DDin / AAin))
						- 0.5 * sqrt(1. + 4. * (l * l * SSin * SSin) * DDin /
							(AAin * AAin * sthin * sthin))
						- 2. * a * rin * l / AAin);
			}
			else
				lnh = 1.;

#if(RAD_M1)
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD] = 0.0;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1_RAD] = ur;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2_RAD] = uh;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3_RAD] = up;
#endif
			/* regions outside torus */
			if (lnh < 0. || r < rin) {
				rho = 0.0; // 1.e-7 * RHOMIN;
				u = 0.0; // 1.e-7 * UUMIN;

				ur = 0.;
				uh = 0.;
				up = 0.;

				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;


#if (DO_YE)
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.5;
#endif
#if (DONUCLEAR)
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 1.0;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 1.0;
#endif			
			}
			/* region inside magnetized torus; u^i is calculated in
			 * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			 * so it needs to be transformed at the end */
			else {
				hm1 = exp(lnh) - 1.;
				rho = pow(hm1 * (gam_local - 1.) / (kappa * gam_local),
					1. / (gam_local - 1.));
				u = kappa * pow(rho, gam_local) / (gam_local - 1.);
				ur = 0.;
				uh = 0.;

				/* calculate u^phi */
				expm2chi = SS * SS * DD / (AA * AA * sth * sth);
				up1 = sqrt((-1. + sqrt(1. + 4. * l * l * expm2chi)) / 2.);
				up = 2. * a * r * sqrt(1. + up1 * up1) / sqrt(AA * SS * DD) +
					sqrt(SS / AA) * up1 / sth;

				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				if (rho > rhomax) {
#pragma omp critical
					rhomax = rho;
				}
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u * (1. + 4.e-2 * (ranc(0) - 0.5));
				if (u > umax && r > rin) {
#pragma omp critical
					umax = u;
				}

#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
				rotate_vector(V, pos_new, &r, &th, &phi, tilt);                  //for rotating contravariant vectors
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V[1];
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V[2];
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
#elif(ELLIPTICAL)
				V_old[1] = ur;
				V_old[2] = uh;
				V_old[3] = up;
				elliptical_vector(X_cart, V_old, V_new, pos_new, &r, &th, eccentricity);
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V_new[1];
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V_new[2];
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V_new[3];

				/* convert from 4-vel to 3-vel */
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
#else
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;//watch out

				/* convert from 4-vel to 3-vel */
				coord_transform(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
#endif
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLR] = 0.0;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLRFRAC] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLR] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][FLRFRAC] = 0.0;

#if (DO_YE)
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.1;
#endif
#if (DONUCLEAR)
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 0.0;
#endif
			}
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][PS1START] = 0.;


			// Setting temperature given u
#if (DOHELM_TEMPERATURE == 2)
			eos_mode_rhou_temp_init(rho, &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], u);
#endif

			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;

			// initialize neutrinos
#if (NEUTRINOS_M1)
			for (int sp = 0; sp < NU_SPECIES; sp++) {
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] = 1e-15;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U1_NU, sp)] = ur;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U2_NU, sp)] = uh;
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(U3_NU, sp)] = up;

				Tnu = pow(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
				p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(NUMBER_NU, sp)] = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tnu);
				//p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][NUMBER_NU] = 1e-30;
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
	if (rank == 0) {
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}

	//double torus_mass = 0.;
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] /= rhomax;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] /= rhomax;

#if(RAD_M1)
			//Set radiation pressure
			init_rad_pres(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
#endif

#if(TWO_T)
			double ue, ui, Theta, C;
			ue = 0.5 * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			if (ue < FLOOR_ENTROPY * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]) {
				ue = FLOOR_ENTROPY * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			ui = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] - ue;

			//Calculate electron entropy
			C = ue / p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
#if(FULL_ENTROPY_VARGAMMA)
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO]);
#else
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRE] = Theta * (Theta + 0.4) / pow(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], 2. / 3.);
#endif

			//Calculate ion entropy
			C = ui / p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
#if(FULL_ENTROPY_VARGAMMA)
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO]);
#else
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][ENTRI] = Theta * (Theta + 0.4) / pow(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], 2. / 3.);
#endif
#endif
		}
	}

	//if (rank == 0) {
		//fprintf(stderr, "torus mass: %g\n", torus_mass);
	//}

	umax /= rhomax;
	rhomax = 1.;

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

}

void set_mag_torus(void) {
	int i, j, z, k, n;
	double rhomax = 1., pmax = 0.;
	int i100 = 0;
	double rho_av, q, beta = BETA, bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM], pos_new[NDIM], beta_ij;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	struct of_state state;
	double gamma_g;

#if(!NSY || CARTESIAN_GR)
	double tilt = (TILT_ANGLE) / 180. * M_PI;
#else
	double tilt = -(TILT_ANGLE) / 180. * M_PI;
#endif	

#if(WHICHPROBLEM==COLLAPSAR)
	double Bfactor = 300; //1e13 G for alpha = 1
	double M_STAR = 14;
	double R_STARcm = 4e10;

	double M_BH = 4;
	double r_rc = M_BH * 1.5e5;
	//double r_rc = R_G_CGS;
	double Rs = R_STARcm / r_rc;
	double Fe_core = 1e8 / r_rc;
	double r_hole = 10;
	double fr;
	beta = 100.0 / (3.6 * Bfactor * Bfactor);
#endif

#if(WHICHPROBLEM==POSTMERGER_PROBLEM && FORNAX_IC)
	double Bfactor = 1e-2; //1e13 G for alpha = 1
	double M_STAR = 7.21;
	double R_STARcm = 2e9;

	double M_BH = M_SGRA_SOLAR;
	double r_rc = R_G_CGS;
	double Rs = R_STARcm / r_rc;
	double Fe_core = 1e8 / r_rc;
	double r_hole = 3.5;
	double fr;
	beta = 100.0 / (3.6 * Bfactor * Bfactor);
#endif

#if(WHICHPROBLEM==BONDI_PROBLEM_2D)
	beta = BETA;
	double rin = R_BONDI;
	double rout = 1e6;
#endif

#if(WHICHPROBLEM==TRUNC_PROBLEM)
	tilt = 0.;
#endif

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + D3) {
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
		}
	}

	/* first find corner-centered vector potential */
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 + D3) {
			/* Cell centered vector potential */
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
#if (TILTED)		
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;
			sph_to_cart(X_cart, &r, &th, &phi);
			rotate_coord(X_cart, -tilt);
			cart_to_sph(X_cart, &r, &th, &phi);
#endif
#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
#elif(WHICHPROBLEM==BONDI_PROBLEM_2D)
			if (r >= rin) q = (r * r - rin * rin) * (sin(th) * sin(th));
			else q = 0.;
#elif(WHICHPROBLEM==COLLAPSAR || (WHICHPROBLEM==POSTMERGER_PROBLEM && FORNAX_IC) || WHICHPROBLEM==NSM)
#if(COLLAPSAR_GR1D)
			q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
#else
			if (r < 1.5 * r_hole) {
				q = r * r * 1e-20 * pow(sin(th), 2);//1 - pow((r-Fe_core)/(Rs-Fe_core),2);
				//q = pow(sin(th),2)*(pow(r_hole,2)/pow(Fe_core,3)-1/Rs)*pow(r/r_hole,2);
			}
			else {
				if (r < Rs) {
					fr = 1 - pow((r - Fe_core) / (Rs - Fe_core), 2);
					// q = pow(sin(th),2)*(pow(sqrt(pow(r,2)-pow(r_hole,2)),2)/(pow(sqrt(pow(r,2)-pow(r_hole,2)),3)+pow(Fe_core,3))-pow(Rs,2)/(pow(Rs,3)+pow(Fe_core,3)));
					q = pow(sin(th), 2) * (pow(r, 2) / (pow(r, 2) + pow(Fe_core, 2)) - pow(r / Rs, 3)); //Fe core profile   
				}
				else {
					fr = 0;
					q = 0;
				}
				// q = fr*pow(sin(th),2)/r; //Komissarov's profile
			}
#endif
#else
			//q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005; //Postmerger problem
			//q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.2; //SANE
			q = pow(p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax, 2.0) * pow(r, 5.0) - 0.2;
			//q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax*pow(r/20.*sin(th),3.)*exp(-r/400.) - 0.2; //code comparison
#endif
			if (q > 0.) {
#if(WHICHPROBLEM==THIN_PROBLEM)
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = q * pow(r, 2.0); //Toroidal

				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = sin(2.0*M_PI *r/120.)*sqrt(r*r*r*r*r)*q;
#else
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = q; //SANE+CODE_COMPARISON
				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = pow(q, 2.0) * pow(r, 3.0); //MAD
#endif
			}
			else {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
			}

#if(WHICHPROBLEM==THIN_PROBLEM)
			double R = r * sin(th);
			double norm_factor = 0.125 * 2.0 * sqrt(7.0 / 0.0007414) * sqrt(0.005608131);
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = norm_factor * (R * (R < 10.0) + 10.0 * (R >= 10.0)); //poloidal field within 10 rg						
#endif

#if (TILTED)
			V[1] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			V[2] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2];
			V[3] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			rotate_vector2(V, pos_new, &r, &th, &phi, tilt);         //for rotating covariant vectors
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = V[1];
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = V[2];
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = V[3];
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
			}
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
			}
			if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
			}
#endif

#if(SPHERICAL || SPHERICAL_GR)
			if (j < 0 || j >= N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= -1.0;
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= -1.0;
			}
#endif

#if(1)
			double dxdxp[NDIM][NDIM], dq_temp[NDIM];
			int k1, k2;
			dxdxp_func(X, dxdxp);

			for (k = 0; k < NDIM; k++) dq_temp[k] = dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k];

			for (k1 = 0; k1 < NDIM; k1++) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k1] = 0;
				for (k2 = 0; k2 < NDIM; k2++) {
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k1] += dxdxp[k2][k1] * dq_temp[k2];
				}
			}
#endif
		}
	}

	//Transform from cell centered vector potential to edge centered vector potential
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - D1, BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] - D2, N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]] - D3, N3_GPU_offset[n_ord[n]] + BS_3) {
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][1] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z - D3)][1]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z - D3)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][2] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z - D3)][2]);
			E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.25 * (dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j - D2, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j, z)][3] + dq[nl[n_ord[n]]][index_3D(n_ord[n], i - D1, j - D2, z)][3]);
		}
	}

	/* now differentiate to find cell-centered B,
	and begin normalization */
#if(STAGGERED)
	gpu = 0;
	nstep = 2 * AMR_SWITCHTIMELEVEL - 1;
	set_prestep();
	const_transport_bound();
	nstep = 0;
#endif
	for (n = 0; n < n_active; n++) {
#if(STAGGERED)
		//Reset toroidal component of vector potential so that no monopoles occur in initial conditions at the pole
		if (block[n_ord[n]][AMR_NBR1] == -1 || block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][3] = 0.;
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = 0.;
			}
		}

		if (block[n_ord[n]][AMR_NBR3] == -1 || block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3) {
			ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][3] = 0.;
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][1] = 0.;
			}
		}

		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			get_geometry(n_ord[n], i, j, z, FACE1, &geom);
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = -(E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][3]) / (dx[nl[n_ord[n]]][2] * geom.g)
#if(N3G>0)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][2]) / (dx[nl[n_ord[n]]][3] * geom.g)
#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE2, &geom);
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][3]) / (dx[nl[n_ord[n]]][1] * geom.g)
#if(N3G>0)
				- (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][1]) / (dx[nl[n_ord[n]]][3] * geom.g)
#endif
				;
			get_geometry(n_ord[n], i, j, z, FACE3, &geom);
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = -(E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][2]) / (dx[nl[n_ord[n]]][1] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][1]) / (dx[nl[n_ord[n]]][2] * geom.g);
		}
#endif
	}

	double bsq_max = 0.;
	double ug_sum = 0.;
	double bsq_sum = 0.;
	for (n = 0; n < n_active; n++) {
		ZLOOP3D_MPI{
			/* flux-ct */
			#if(!STAGGERED)
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B1] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2. * dx[nl[n_ord[n]]][2] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				+ E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2. * dx[nl[n_ord[n]]][3] * geom.g);
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B2] =
				(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][3]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][3] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j + 1, z)][3]) / (2. * dx[nl[n_ord[n]]][1] * geom.g)
				- (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j + 1, z)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + 1)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2. * dx[nl[n_ord[n]]][3] * geom.g);
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B3] =
				-(E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][2]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z)][2] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i + 1, j, z + 1)][2]) / (2. * dx[nl[n_ord[n]]][1] * geom.g)
				+ (E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] + E_corn[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z + 1)][1]
				- E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z)][1] - E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, j + 1, z + 1)][1]) / (2. * dx[nl[n_ord[n]]][2] * geom.g);
			#else
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = (ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE1] + ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i + D1, j, z)][1] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i + D1, j, z)][FACE1]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = (ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE2] + ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j + D2, z)][2] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j + D2, z)][FACE2]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#if(N3G>0)
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = (ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][FACE3] + ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z + D3)][3] * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z + D3)][FACE3]) / (2.0 * gdet[nl[n_ord[n]]][index_2D(n_ord[n], i, j, z)][CENT]);
			#else
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
			#endif
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#endif
			bsq_ij = bsq_calc(p_torus[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);

			beta_ij = 2.0 * (gam - 1.0) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] / bsq_ij;
			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0 / 3.0;
				#else
				gamma_g = GAMMA;
				#endif			
			#else
			gamma_g = GAMMA;
			#endif

			#if(WHICHPROBLEM==COLLAPSAR || WHICHPROBLEM==NSM)
			coord(n_ord[n], i - 2, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (r < Rs) {
				coord(n_ord[n], i + 2, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				//if (r > Rs) bsq_ij = 0.;
			}
			coord(n_ord[n], i - 2, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (r < r_hole) {
				coord(n_ord[n], i + 2, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				//if (r > r_hole) bsq_ij = 0.;
			}
			#endif

			#if(RAD_M1)
			if (((gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 4) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				pmax = (gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else

			if ((gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && j > (int)(20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && j < (int)(N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				bsq_max = bsq_ij;
			}
			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r < 50.) {
				bsq_sum += bsq_ij * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
#endif
#endif

	if (rank == 0) {
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}

	/* finally, normalize to set field strength */
#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.) * ug_sum / (0.5 * bsq_sum);
	if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	norm = sqrt(0.0007414 / 7.0);
#else
	beta_act = pmax / (0.5 * bsq_max);
	if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	norm = sqrt(beta_act / beta);
#endif

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] *= norm;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] *= norm;
			p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] *= norm;
#if(STAGGERED)
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] *= norm;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= norm;
			ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= norm;
#endif
		}
	}

#if(RESISTIVE)
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + D3) {
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			set_E_init(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], geom);
		}
	}
#endif

	bsq_max = 0.;
	pmax = 0;
	bsq_sum = 0.;
	ug_sum = 0.;
	for (n = 0; n < n_active; n++) {
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p_torus[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);

			#if(WHICHPROBLEM==COLLAPSAR || WHICHPROBLEM==NSM)
			coord(n_ord[n], i - 2, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (r < Rs) {
				coord(n_ord[n], i + 2, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				if (r > Rs) bsq_ij = 0.;
			}
			coord(n_ord[n], i - 2, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (r < r_hole) {
				coord(n_ord[n], i + 2, j, z, CENT, X);
				bl_coord(X, &r, &th, &phi);
				if (r > r_hole) bsq_ij = 0.;
			}

			#endif

			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				bsq_max = bsq_ij;
			}

			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0 / 3.0;
				#else
				gamma_g = GAMMA;
				#endif
			#else
			gamma_g = GAMMA;
			#endif

			#if(RAD_M1)
			if (((gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else
			if ((gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r < 50.) {
				bsq_sum += bsq_ij * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

	/*Share bsq_max among MPI processes*/
#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
#endif
#endif

#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.) * ug_sum / (0.5 * bsq_sum);
#else
	beta_act = pmax / (0.5 * bsq_max);
#endif
	if (rank == 0) {
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1 + D3) {
#if(ANTIPARALLEL_FIELDS)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] -= p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] -= p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] -= p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] -= ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] -= ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] -= ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] -= ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
#else
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2];
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] += p_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] += ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] += ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] += ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2];
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] += ps_torus[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3];
#endif	
		}
	}
}
#endif

double f_misalignment(double r)
{
	double r0, r1; // radii between which to realign
	double f; // 1 inside r0, 0 outside r1
	r0 = 4.5*R_NS/4.0;
	r1 = 8.5*R_NS/4.0;

	if (r < r0)
		f = 1.0;
	else if (r > r1)
		f = 0.0;
	else
		f = 0.5 * (1.0 + cos(M_PI * (r - r0) / (r1 - r0)));

	return f;
}



