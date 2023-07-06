#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

#undef dd
#define dd(ii,ivar) icdata[ivar*nx+ii]

void init_collapsar(void)
{
	int interpolate_gr1d_prims(double r, double th, double ph, extent1d ext, double* data, double* p);
	char* read_first_line(char* s, size_t size, FILE * fp);
	char* read_last_line(char* s, size_t size, FILE * fp);
	int i, j, z, n;
	double r, th, phi, sth, cth;
	double ur, uh, up, u, rho;
	double bl_gcov[NDIM][NDIM];
	double X[NDIM], X_cart[NDIM], V[NDIM], V_old[NDIM], V_new[NDIM], pos_new[NDIM];
	double rhor, M_STAR, M_BH, Rs, alphap, betap, rho0, R_STARcm, Omega0, Omega0_limit, r_rc, t_rc, m_rc, r_hole;
	struct of_geom geom;

	/* for disk interior */
	double DD, AA, SS, thin, sthin, cthin, DDin, AAin, SSin;
	double l, rin, lnh, expm2chi, up1;
	double tilt, eccentricity;
	double kappa, hm1;

	/*For MPI*/
	double inmsg;

	// for magnetic field
	double rho_av, rhomax, umax, beta, bsq_ij, bsq_max, norm, q, beta_act;
	double rhofactor = 1e3;
	// star parameters
	M_STAR = 14; // Stellar mass
	R_STARcm = 4e10; // Star radius
	M_BH = 4; // Stellar mass
	r_rc = M_BH * 1.5e5;
	//r_rc = R_G_CGS;
	m_rc = M_BH * 2e33;
	Rs = R_STARcm / r_rc; // stellar radius in code units
	alphap = 1.5; // inner density profile power-law
	betap = 3; // outer density profile power-law
	rho0 = 0.3; // density normalization for alphap = 1
	Omega0 = 5;
	Omega0_limit = 1e-3;
	r_hole = 10;

	double temp = a;
	a = 0.8;
	rin = 6.0;
	rmax = 1e4;
	l = lfish_calc(rmax);
	kappa = 1.e-3;
	beta = 100.;
	
	int res;
	#if (COLLAPSAR_GR1D)
	FILE* fp1;
	int ind;
	int nitems_read, nitems_expected;
	# define MAXLEN (1024)
	int nvars, nx, ny, nz;
	double* icdata;
	extent1d ext;

	char fname1[] = "GR1D_star.dat";

	char first_line[MAXLEN], last_line[MAXLEN], buf1[MAXLEN], * ptr1;
	size_t memsize, nitems, nread;
	double prim[NPR];
	int k, ii, jj, kk;

	ext.nvars = 6;

	for (ind = 0; ind < numtasks; ind++) {
		if (ind == rank) {
			fp1 = fopen(fname1, "rb");
			if (NULL == fp1 && 0 == rank) {
				fprintf(stderr, "Could not open file %s for reading, exiting\n", fname1);
				exit(1234);
			}
			ext.nx = 1003;

			read_last_line(last_line, MAXLEN, fp1);
			sscanf(last_line, "%lf %*lf %*lf %*lf %*lf", &ext.xmax);
			rewind(fp1);

			read_first_line(first_line, MAXLEN, fp1);
			sscanf(first_line, "%lf %*lf %*lf %*lf %*lf", &ext.xmin);

			if (0 == rank) {
				fprintf(stderr, "[%d] reading IC block: resolution (%dx%d), extent (%g,%g), files %s...", rank, ext.nvars, ext.nx, ext.xmin, ext.xmax, fname1);
				fflush(stderr);
			}

			nx = ext.nx;
			nvars = ext.nvars;
			nitems = (size_t)nvars * nx;
			memsize = sizeof(double) * nitems;
			icdata = malloc(memsize);

			if (NULL == icdata) {
				fprintf(stderr, "[%5d] could not allocate memory of size %ld\n", rank, memsize);
				fclose(fp1);
				exit(1235);
			}

			for (ii = 0; ii < nx; ii++) {

				//first file, containing grid and data information
				ptr1 = fgets(buf1, MAXLEN, fp1);
				if (NULL == ptr1) break;

				dd(ii, 0) = (double)ii;

				nitems_read = sscanf(ptr1, "%lf %lf %lf %lf %lf \n", &dd(ii, 1), &dd(ii, 2), &dd(ii, 3), &dd(ii, 4), &dd(ii, 5));

				nitems_expected = 5;
				if (nitems_expected != nitems_read) break;
			}

			if (nitems_expected != nitems_read || ferror(fp1) || (NULL == ptr1 && !feof(fp1))) {
				fprintf(stderr, "[%5d] Error reading from file(s)\n", rank);
			}
			fclose(fp1); fp1 = NULL;
			

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
	#endif
	
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

	// output choices
	tf = 200000000.0;

	// start diagnostic counters
	dump_cnt = 0;
	dump_cnt_reduced = 0;
	image_cnt = 0;
	rdump_cnt = 0;

	rhomax = 0.;
	umax = 0.;
	#if(!NSY)
	tilt = (TILT_ANGLE) / 180. * M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180. * M_PI;
	#endif
	eccentricity = 0;

	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z, res) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa,hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
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

			#if (COLLAPSAR_GR1D)
			res = interpolate_gr1d_prims(r, th, phi, ext, icdata, p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);

			if (res) {
				//|| p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] * MASS_DENSITY_SCALE > 1e13) {
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = 1e-7 * RHOMIN;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1e-7 * UUMIN;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = 0.0;

				#if (DO_YE)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.5;
				#endif
				#if (DONUCLEAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 1.0;
				#endif
			}
			else {
				/* convert from BL 4-vel to relative 4-vel in internal (KS prime) coords */
				#if (DONUCLEAR)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA] = 0.0;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM] = 0.0;
				#endif
			}

			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;

			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] > rhomax) {
				#pragma omp critical
				rhomax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO];
			}
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > umax && r > rin) {
				#pragma omp critical
				umax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#else
			ur = 0.;
			uh = 0.;
			up = 0.;
			if (r > Rs) {
				rho = 1e-20; // ISM density
				u = rho / 1e6; //u = 1e-3*pow(rho,gam)/(gam - 1.) ;

				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;
			}
			else if (r < 2) {
				rho = rho0 * pow(r_hole, -alphap) * r / r_hole;
				//rho = rho0 * pow(r_hole, -alphap);
				u = 1e-6 * rho / r;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;
			}
			else {
				if (r < r_hole) {
					rho = rho0 * pow(r_hole, -alphap) * r / r_hole;
					u = 1e-6 * rho / r;
					/* p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][RHO] = rho;
					p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][UU] = u;
					p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U1] = ur;
					p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U2] = uh;
					p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U3] = up;*/
				}
				else {
					rho = rho0 * pow(r, -alphap) * pow((Rs - r) / Rs, betap);
					u = 1e-6 * rho / r;
					//u = 0.5 * rho / (r * (gam - 1.)) * M_BH * M_BH / M_STAR;
				}
				//u = 1e-3*pow(rho,gam)/(gam - 1.) ;
				//u = 1/(gam-1)*3.14*pow(rho0,2)/pow(Rs,3) * (1/(12*pow(r*Rs,3)) * (3*pow(r,7)-28*pow(r,6)*Rs+126*pow(r,5)*Rs*Rs-420*pow(r,4)*pow(Rs,3)+252*r*r*pow(Rs,5)-42*r*pow(Rs,6)+4*pow(Rs,7)) - 35*Rs*log(r)); // Newtonian hydrostatic eq.
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] = rho;
				if (rho > rhomax) {
					#pragma omp critical
					rhomax = rho;
				}
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = u * (1. + 4.e-2 * (ranc(0) - 0.5));
				if (u > umax && r > rin) {
					#pragma omp critical
					umax = u;
				}
				//up = Omega0/(1+pow(r/A_rot,2))/3e10; // angular velocity
				//if (Omega0/pow(r*sin(th),2) > Omega0_limit) {
				if (Omega0 / pow(r, 2) > Omega0_limit) {
					up = Omega0_limit;
				}
				else {
					//up = Omega0/pow(r*sin(th),2);
					up = Omega0 / pow(r, 2);
				}
				#if (TILTED)
				V[1] = ur;
				V[2] = uh;
				V[3] = up;
				rotate_vector(V, pos_new, &r, &th, &phi, tilt);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V[3];

				// convert from 4-vel to 3-vel
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#elif(ELLIPTICAL)
				V_old[1] = ur;
				V_old[2] = uh;
				V_old[3] = up;
				elliptical_vector(X_cart, V_old, V_new, pos_new, &r, &th, eccentricity);
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = V_new[1];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = V_new[2];
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = V_new[3];

				// convert from 4-vel to 3-vel
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#else
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U1] = ur;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U2] = uh;
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][U3] = up;//watch out

				// convert from 4-vel to 3-vel
				coord_transform(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)], n_ord[n], i, j, z);
				#endif
			}
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][B3] = 0.;
			#if (DO_YE)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] = 0.5;
			#endif
			#endif
		}
	}
	a = temp;
	#if (MPI_enable)
	// Share rhomax among MPI processes
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);

	// Share umax among MPI processes
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#endif

	// Normalize the densities so that max(rho) = 1
	if (rank == 0) {
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}
	//ZSLOOP(0,N1-1,0,N2-1) {
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			#if (COLLAPSAR_GR1D == 0)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] /= rhofactor * rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] /= rhofactor * rhomax;
			#endif
		}
	}
	umax /= rhomax;
	rhomax = 1.;
	/*        for (n = 0; n < n_active; n++) ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]], N2_GPU_offset[ \
		n_ord[n]] + BS_2, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3) {
		  dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][0] = 0.;
		  dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.;
		  dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.;
		  dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.;
		}
	*/
	for (n = 0; n < n_active; n++) {
		fixup(p, n_ord[n]);
	}

	bound_prim(p, 1);

	//set_mag();


	#if (DOHELM)
	// Using density and pressure = (gam - 1) * u, find new u, using Helmholtz EOS
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			#if (1)
			//#if (COLLAPSAR_GR1D)
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] *= (gam - 1.);
			eos_mode_rhopres_u(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
			#endif
		}
	}

	//for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);

	#if (DOHELM_TEMPERATURE)
	// Set temperatures given u:
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			#if (0)
			//#if (COLLAPSAR_GR1D == 0)
			//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1e-6;
			#else
			eos_mode_rhou_temp_init(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]
				#if (DONUCLEAR)
				, &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA], &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM]
				#endif
			);
			//p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] = 1e3;
			#endif
		}
	}
	#endif
	#endif

	#if(NEUTRINOS_M1)
	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			//init_nuclear(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
			init_neutrinos(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}
	#endif

	sourceflag = 0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif

	bound_prim(p, 1);

}

int interpolate_gr1d_prims(double r, double th, double ph, extent1d ext, double* data, double* p)
{
	int interpolate_gr1d_var(double r, double th, double ph, extent1d ext, double* data, int ivar, double* val);
	int res;

	res = interpolate_gr1d_var(r, th, ph, ext, data, 2, &p[RHO]);
	res += interpolate_gr1d_var(r, th, ph, ext, data, 3, &p[UU]); 
	p[UU] /= (gam - 1);
	#if(DO_YE)
	res += interpolate_gr1d_var(r,th,ph,ext,data,4,&p[YE]);
	#endif
	//res += interpolate_gr1d_var(r, th, ph, ext, data,5,&p[U1]);

	p[U1] = 0.;
	p[U2] = 0.;
	p[U3] = 0.;
	p[B1] = 0.;
	p[B2] = 0.;
	p[B3] = 0.;

	return(res);
}

//define compact form for array indexing
#define d(ii) data[ivar*nx+ii]

int interpolate_gr1d_var(double r, double th, double ph, extent1d ext, double* data, int ivar, double* val)
{
	double x, dx;
	double i, di;
	int i0, i1, nx;
	double c;

	nx = ext.nx;
	x = log(r);
	
	// find value of r from input file
	i0 = 0;
	while (r > data[1 * nx + i0] && i0 < nx) {
		i0++;
	}
	i0--;
	i1 = i0 + 1;

	if (i1 >= nx) {
		return(1);
	}

	dx = log(data[1 * nx + i1]) - log(data[1 * nx + i0]);
	di = (x - log(data[1 * nx + i0])) / dx;
	
	c = log(d(i0))* (1 - di) + log(d(i1)) * di;
	if (isnan(c))  return(1);
	*val = exp(c);
	return(0);
}