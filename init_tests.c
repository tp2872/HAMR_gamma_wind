#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

void init_blastwave()
{
	int n, i, j, z, k;
	double xx, yy, zz, r, th, phi, dist, X[NDIM];
	double x0, y0, z0, radius, scale_factor;
	int do_mag;
	struct of_geom geom;

	/* some physics parameters */
	gam = GAMMA;

	/* some numerical parameters */
	failed = 0;	/* start slow */
	dt = 1.e-5;
	t = 0.;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	// Override tf and the dump and log intervals
	tf = 200000.0;

	//Set central coordinates of exploding ball and radius
	x0 = 5.0;
	y0 = 0.0;
	z0 = 10.0;
	radius = 2.0;

	//Decide if magnetic field is enabled
	do_mag = 1;

	//Set scale factor to not conflict with density floors
	scale_factor = 1.0e4;

	//Initialize magnetic arrays
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
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
		}
	}

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);

			//Calculate coordinates of point (i,j,k)
			xx = r * sin(th) * cos(phi);
			yy = r * sin(th) * sin(phi);
			zz = r * cos(th);

			//Calculate distance to center of explosion
			dist = sqrt(pow(xx - x0, 2.0) + pow(yy - y0, 2.0) + pow(zz - z0, 2.0));
			double dist2 = sqrt(pow(r * sin(th) - sqrt(x0 * x0 + y0 * y0), 2.0) + pow(zz - z0, 2.0));

			//Set ug_norm to insert toroidal magnetic field
			double ug_norm = 3.0e-5 / (GAMMA - 1.0) * scale_factor;
			scale_factor = 1.0e4;

			if (dist < radius) {
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0e-4 * scale_factor;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = ug_norm;
			}
			else {
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0e-4 * scale_factor * exp(-4.0*fabs(dist-radius));
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 3.0e-5 / (GAMMA - 1.0) * scale_factor * exp(-4.0 * fabs(dist - radius));
			}

			if (dist < 0.8*radius) {
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = scale_factor*1.0e-2;
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = scale_factor * 1.0e-1 / (GAMMA - 1.0);
			}
			else if (dist < radius) {
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = scale_factor * 1.0e-2*exp(-log(1e2) / (0.2 * radius)*fabs(dist - 0.8 * radius));
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = scale_factor * 1.0e-1 / (GAMMA - 1.0) * exp(log(3e-5) / (0.2 * radius) * fabs(dist - 0.8 * radius));
			}
			else {
				//scale_factor = 1.0e-2;
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = scale_factor * 1.0e-4;
				//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = scale_factor * 3.0e-6 / (GAMMA - 1.0) * exp(-4.0 * fabs(dist - radius));
			}

			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;

			if (do_mag == 1 && dist<radius) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = fabs(radius-dist);

				/*ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 1.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 1.0;
				get_geometry(n_ord[n], i, j, z, CENT, &geom);
				double bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom);
				double beta = (GAMMA - 1.0) * ug_norm / (0.5 * bsq_ij);
				double beta_target = 100.0;
				double norm = beta_target / beta;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] /= norm;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] /= norm;
				*/
			}
			else {
				#if(STAGGERED)
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
				ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
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
	nstep = 2*AMR_SWITCHTIMELEVEL - 1;
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
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]] + BS_2, z)][1] = 0.;
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

	double bsq_ij, beta_ij, gamma_g, beta_act, norm;
	double bsq_max = 0.;
	double ug_sum = 0.;
	double bsq_sum = 0.;
	double pmax = 0.;
	double beta = 10.;
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
			//if (bsq_ij > 0.0)fprintf(stderr, "Test: %f \n", log10(bsq_ij));
			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0/3.0;
				#else
				gamma_g = GAMMA;
				#endif			
			#else
			gamma_g = GAMMA;
			#endif

			if ((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && (j > 4) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}

			if (bsq_ij > bsq_max && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				bsq_max = bsq_ij;
			}

		}
	}

	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);

	/* finally, normalize to set field strength */
	beta_act = pmax / (0.5*bsq_max);
	if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	norm = sqrt(beta_act / beta);
	if (do_mag == 0) norm = 1.0;

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

	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);

	/* enforce boundary conditions */
	bound_prim(p, 1);
}

void init_radpulse()
{
	int i, j, k, z, n;
	double x, y, zz, sth, cth;
	double ur, uh, up, u, rho;
	double X[NDIM];
	struct of_geom geom;

	double sigma = 1.54e-54; // 8.77e-12 * 0.25 * C_CGS * (ENERGY_DENSITY_SCALE / pow(MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.));
	double T0 = 1e6;
	double myrho = 1.;
	double xc = 0.;
	double yc = 0.;
	double zc = 0.;
	double w = 5.;
	double T_rad;
	double tau = 0.;
	double taumax = 0.;

	/* some physics parameters */
	gam = GAMMA;

	/* some numerical parameters */
	lim = MC;
	failed = 0;	/* start slow */
	dt = 1.e-5;

	t = 0.; 

	/* output choices */
	tf = 1000.;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &x, &y, &zz);
			//applying the perturbations
			T_rad = T0 * (1. + 100. * exp(- ((x - xc) * (x - xc) + (y - yc) * (y - yc) + (zz - zc) * (zz - zc)) / (w * w)));
			//T_rad = T0 * (100. * exp(- ((x - xc) * (x - xc)) / (w * w)));
			//T_rad = T0 * (1. + 100. * exp(-((x - xc) * (x - xc)) / (w * w)));

			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = myrho;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = myrho * T0 / (GAMMA - 1.);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;

#if(RAD_M1)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD] = 4. * sigma * pow(T_rad, 4.0);
			//fprintf(stderr, "%e\n", p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1_RAD] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2_RAD] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3_RAD] = 0.;
#endif
		}
	}

	double cell_size, kappa_es, kappa_abs;
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			//Calculate optical depth of one cell
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			#if(D3>1)
			cell_size = MY_MAX(MY_MAX(dx[nl[n_ord[n]]][1] * sqrt(geom.gcov[1][1]), dx[nl[n_ord[n]]][2] * sqrt(geom.gcov[2][2])), dx[nl[n_ord[n]]][3] * sqrt(geom.gcov[3][3]));
			#else
			cell_size = MY_MAX(dx[nl[n_ord[n]]][1] * sqrt(geom.gcov[1][1]), dx[nl[n_ord[n]]][2] * sqrt(geom.gcov[2][2]));
			#endif
			//Calculate radiation temperature in rest frame of fluid
			struct of_state q;
			struct of_state_rad q_rad;
			get_state(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom, &q);
			get_state_rad(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], &geom, &q_rad);
			double Tr = calc_Tr(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], q.ucon, q_rad.ucon, q_rad.ucov);
			double 	bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
			double gamma_g = GAMMA;
			kappa_abs = calc_kappa_abs(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], bsq, Tr
				#if(TWO_T)
				, gamma_g
				#endif
			);
			kappa_es = calc_kappa_es(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]
				#if(TWO_T)
				, gamma_g
				#endif
			);
			tau = (kappa_es + kappa_abs) * cell_size;
			if (tau > taumax) {
				taumax = tau;
			}
		}
	}

	fprintf(stderr, "sigma=%e, taumax=%e\n", sigma, taumax);

	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++) {
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
}

void init_shocktube()
{
	int n, i, j, z, k;
	double xx, yy, zz, r, th, phi, dist, X[NDIM];
	double x0, y0, z0, radius, scale_factor;
	int mode;
	struct of_geom geom;

	/* some physics parameters */
	gam = GAMMA;

	/* some numerical parameters */
	failed = 0;	/* start slow */
	dt = 1.e-5;
	t = 0.;
	tf = 2.0;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	/*Set mode
	fast shock (1), slow shock (2), Switch off fast (3), Switch on slow (4), Alfven wave (5)
	Compound wave (6), Shock tube 1 (7), Shock tube 2 (8), Collision (9)
	*/
	mode = 1; 

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);

			if (mode==1) {//Fast shock
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 25.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 20.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 25.02;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 20.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 25.02;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 25.48;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 367.5 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 1.091;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.3923;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 20.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 49.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 20.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 49.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 2.5;
			}
			if (mode==2) {//Slow shock
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 10.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 1.53;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 18.28;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 18.28;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 3.323;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 55.36 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.9571;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = -0.6822;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 14.49;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 14.49;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 2.0;
			}

			if (mode==3) {//Switch off fast
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.1;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = -2.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 2.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 2.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.562;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 10.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = -0.212;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = -0.590;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 2.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 4.71;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 2.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 4.71;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 1.0;
			}

			if (mode==4) {//Switch on slow
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.00178;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 0.1 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = -0.765;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = -1.386;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 1.022;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 1.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 1.022;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.01;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 1.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 2.0;
			}

			if (mode==5) {//Alfven wave
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 3.7;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 5.76;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = -6.857;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = -6.857;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 2.0;
			}

			if (mode==6) {//Compound wave
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 3.7;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 5.76;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 3.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = -6.857;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 3.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = -6.857;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 1.5;
			}

			if (mode==7) {//Shock tube 1
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1000.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 1.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.1;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 1.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 1.0;
			}

			if (mode==8) {//Shock tube 2
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 30.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 20.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 20.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 0.1;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 1.0;
			}
			if (mode==9) {//Collision
				if (X[1] < 0.0) {//left state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 5.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				else {//right state
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1.0 / (GAMMA - 1.0);
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = -5.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = -10.0;
					p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.0;
					#if(STAGGERED)
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = -10.0;
					ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
					#endif
				}
				tf = 1.22;
			}
		}
	}

	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);

	//Set constant boundary conditions
	#if(CONSTANT_BC)
	int i1;
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

	/* enforce boundary conditions */
	bound_prim(p, 1);
}

void init_entwave()
{
	int n, i, j, z, k;
	struct of_geom geom;

	/* some physics parameters */
	gam = GAMMA;

	/* some numerical parameters */
	failed = 0;	/* start slow */
	dt = 1.e-5;
	t = 0.;

	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	double X[NDIM];

	// Mean state
	double rho0 = 1.;
	double u0 = 1.; // TODO set U{n} for boosted entropy
	double B10 = 0.; // This is set later, see below
	double B20 = 0.;
	double B30 = 0.;

	// Wavevector
	double k1 = 2. * M_PI;
	double k2 = 2. * M_PI;
	double k3 = 2. * M_PI;
	double amp = 1.e-5;

	// "Faux-2D" planar waves direction
	// Set to 0 for "full" 3D wave
	int dir = 0;
	int nmode = 3;

	double omega, drho, du, du1, du2, du3, dB1, dB2, dB3;

	// Default value 0
	omega = 0.;
	drho = 0.;
	du = 0.;
	du1 = 0.;
	du2 = 0.;
	du3 = 0.;
	dB1 = 0.;
	dB2 = 0.;
	dB3 = 0.;

	if (dir == 1)
		k1 = 0;
	if (dir == 2)
		k2 = 0;
	if (dir == 3)
		k3 = 0;

	// "Faux-2D" planar waves direction
   // Set to 0 for "full" 3D wave
	if (dir == 0)
	{
		// 3D (1,1,1) wave
		B10 = 1.;
		if (nmode == 0)
		{ // Entropy
			omega = 2. * M_PI / 5. * 1;
			drho = 1.;
		}
		else if (nmode == 1)
		{ // Slow
			omega = 2.35896379113;
			drho = 0.556500332363;
			du = 0.742000443151;
			du1 = -0.282334999306;
			du2 = 0.0367010491491;
			du3 = 0.0367010491491;
			dB1 = -0.195509141461;
			dB2 = 0.0977545707307;
			dB3 = 0.0977545707307;
		}
		else if (nmode == 2)
		{ // Alfven
			omega = -3.44144232573;
			du2 = -0.339683110243;
			du3 = 0.339683110243;
			dB2 = 0.620173672946;
			dB3 = -0.620173672946;
		}
		else
		{ // Fast
			omega = 6.92915162882;
			drho = 0.481846076323;
			du = 0.642461435098;
			du1 = -0.0832240462505;
			du2 = -0.224080007379;
			du3 = -0.224080007379;
			dB1 = 0.406380545676;
			dB2 = -0.203190272838;
			dB3 = -0.203190272838;
		}
	}
	else
	{
		// 2D (1,1,0), (1,0,1), (0,1,1) wave
		// Constant field direction
		if (dir == 1)
		{
			B20 = 1.;
		}
		else if (dir == 2)
		{
			B30 = 1.;
		}
		else if (dir == 3)
		{
			B10 = 1.;
		}

		if (nmode == 0)
		{ // Entropy
			omega = 2. * M_PI / 5. * 1;
			drho = 1.;
		}
		else if (nmode == 1)
		{ // Slow
			omega = 2.41024185339;
			drho = 0.558104461559;
			du = 0.744139282078;
			if (dir == 1)
			{
				du2 = -0.277124827421;
				du3 = 0.0630348927707;
				dB2 = -0.164323721928;
				dB3 = 0.164323721928;
			}
			else if (dir == 2)
			{
				du3 = -0.277124827421;
				du1 = 0.0630348927707;
				dB3 = -0.164323721928;
				dB1 = 0.164323721928;
			}
			else if (dir == 3)
			{
				du1 = -0.277124827421;
				du2 = 0.0630348927707;
				dB1 = -0.164323721928;
				dB2 = 0.164323721928;
			}
		}
		else if (nmode == 2)
		{ // Alfven
			omega = 3.44144232573;
			if (dir == 1)
			{
				du1 = 0.480384461415;
				dB1 = 0.877058019307;
			}
			else if (dir == 2)
			{
				du2 = 0.480384461415;
				dB2 = 0.877058019307;
			}
			else if (dir == 3)
			{
				du3 = 0.480384461415;
				dB3 = 0.877058019307;
			}
		}
		else
		{ // Fast
			omega = 5.53726217331;
			drho = 0.476395427447;
			du = 0.635193903263;
			if (dir == 1)
			{
				du2 = -0.102965815319;
				du3 = -0.316873207561;
				dB2 = 0.359559114174;
				dB3 = -0.359559114174;
			}
			else if (dir == 2)
			{
				du3 = -0.102965815319;
				du1 = -0.316873207561;
				dB3 = 0.359559114174;
				dB1 = -0.359559114174;
			}
			else if (dir == 3)
			{
				du1 = -0.102965815319;
				du2 = -0.316873207561;
				dB1 = 0.359559114174;
				dB2 = -0.359559114174;
			}
		}
	}


	// Override tf and the dump and log intervals
	tf = 2. * M_PI / fabs(omega);
	fprintf(stderr,"tf: %f \"n", tf);

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]] - N1G, BS_1 + N1_GPU_offset[n_ord[n]] - 1 + N1G, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1 + N3G) {
			coord(n_ord[n], i, j, z, CENT, X);

			double mode;
			//if (dir == 1) mode = amp * cos(k2 * X[2] + k3 * X[3]);
			//else if (dir == 2) mode = amp * cos(k1 * X[1] + k3 * X[3]);
			//else if (dir == 3) mode = amp * cos(k1 * X[1] + k3 * X[2]);
			//else 
				mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);

			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho0 + drho * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u0 + du * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = du1 * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = du2 * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = du3 * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = B10 + dB1 * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = B20 + dB2 * mode;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = B30 + dB3 * mode;

			#if(STAGGERED)
			coord(n_ord[n], i, j, z, FACE1, X);
			mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = B10 + dB1 * mode;

			coord(n_ord[n], i, j, z, FACE2, X);
			mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = B20 + dB2 * mode;

			coord(n_ord[n], i, j, z, FACE3, X);
			mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = B30 + dB3 * mode;
			#endif
		}
	}

	/* enforce boundary conditions */
	bound_prim(p, 1);
}

void init_sndwave()
{
	int i, j, k, z, n;
	double x, y, zz, sth, cth;
	double ur, uh, up, u, rho;
	double X[NDIM];
	struct of_geom geom;

	double myrho, myu, mycs, myv;
	double delta_rho;
	double cosa, sina;
	double delta_ampl = 1e-5; //amplitude of the wave
	double k_vec_x = 2 * M_PI;  //wavevector
	double k_vec_y = 2 * M_PI;
	double k_vec_len = sqrt(k_vec_x * k_vec_x + k_vec_y * k_vec_y);
	double tfac = 1e3; //factor by which to reduce velocity

	/* some physics parameters */
	gam = GAMMA;

	/* some numerical parameters */
	lim = MC;
	failed = 0;	/* start slow */
	dt = 1.e-5;

	t = 0.;


	myrho = 1.;
	myu = myrho / (gam * (gam - 1));  //so that mycs is unity

	mycs = sqrt(gam * (gam - 1) * myu / myrho);  //background sound speed

	/* output choices */
	tf = tfac / mycs*sqrt(2);
	fprintf(stderr, "tf: %f \"n", tf);


	/* start diagnostic counters */
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;
	defcon = 1.;

	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]]-N1G, BS_1 + N1_GPU_offset[n_ord[n]] - 1 + N1G, N2_GPU_offset[n_ord[n]] -N2G, N2_GPU_offset[n_ord[n]] + BS_2 - 1 + N2G, N3_GPU_offset[n_ord[n]]-N3G, N3_GPU_offset[n_ord[n]] + BS_3 - 1+N3G){
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &x, &y, &zz);
			//applying the perturbations
			delta_rho = delta_ampl * cos(k_vec_x * x + k_vec_y * y);

 			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = myrho + delta_rho;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = (myu + gam * myu * delta_rho / myrho) / (tfac * tfac);
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = (delta_rho / myrho * mycs * k_vec_x / k_vec_len) / tfac;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = (delta_rho / myrho * mycs * k_vec_y / k_vec_len) / tfac;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
			ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
		}
	}

	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++) {
		fixup(p, n_ord[n]);
	}
	bound_prim(p,1);
}