#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

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
				#if(RAD_M1)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD] = pow(10., -300.);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1_RAD] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2_RAD] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3_RAD] = up;
				#endif
			}
			/* region inside magnetized torus; u^i is calculated in
			* Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			* so it needs to be transformed at the end */
			else {
				up = 1. / (pow(r, 3. / 2.) + a);
				up *= sqrt(1. / (1 - up*up*r*r));
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

			#if(RAD_M1)
			init_rad_pres(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
			#endif
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
	bound_prim(p, 1, t);

	set_mag();

	sourceflag = 0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif
}