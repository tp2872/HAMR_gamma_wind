#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

#define dd(ii,jj,kk,ivar) icdata[((ivar*nx+ii)*ny+jj)*nz+kk]
#define VARI 0
#define VARJ 1
#define VARK 2
#define VARR 3
#define VARTHETA 4
#define VARPHI 5
#define VARRHO 6
#define VARP 7
#define VARYE 8
#define VARMUDT 9
#define VARUDPHI 10 // Added udphi to adjust Utilde^phi
#define VARVUR 11
#define VARVUTHETA 12
#define VARVUPHI 13
#define NVARS 14

void init_postmerger() {
	int interpolate_spec_prims( double r, double th, double ph, extent ext, double *data, double *p, double* udphi, double* mudt);
	char* read_first_line(char *s, size_t size, FILE *fp);
	char* read_last_line(char *s, size_t size, FILE *fp);
	int i,j,z,n ;
	extent ext;
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
	double r_unit = M_SGRA_SOLAR; //conversion factor = (Mbh/Msun)

	FILE *fp1, *fp2;
	int ind;
	int nitems_read, nitems_expected;
	# define MAXLEN (1024)
	int nvars, nx, ny, nz;
	int res;
	double *icdata;

	#if (BHNSQ2)
	#if (BHNSQ2_1)
	char fname1[] = "spec_ic_1.dat";
	#else
	char fname1[] = "InterpolatedDataBHNSQ2.dat";
	#endif
	#else
	char fname1[] = "PointsToInterpolateHAMR.dat";
	char fname2[] = "HARM_DataWithMap_27Jul2018.dat";
	#endif

	// In case you want to read the whole ICs table -- set all of them to 1.
	// Initial resolution is 512 x 256 x 128
	int stride1 = 1;
	int stride2 = 1; 
	int stride3 = 1;

	char first_line[MAXLEN], last_line[MAXLEN], buf1[MAXLEN], buf2[MAXLEN], buf3[MAXLEN], *ptr1, *ptr2;
	size_t memsize, nitems, nread;
	double prim[NPR];
	int k, ii, jj, kk;
	double udphi, mudt;

	/* disk parameters (use fishbone.m to select new solutions) */
	a = BH_SPIN ;
	beta = BETA;

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
	tf = 200000000.0 ;

	/* start diagnostic counters */
	dump_cnt = 0 ;
	image_cnt = 0 ;
	rdump_cnt = 0 ;

	ext.nvars = NVARS;
	//read ICs from file
	//for this, loop over all MPI processes
	//and let them read the IC data from file, one by one

	for (ind = 0; ind < numtasks; ind++) {
		if (ind == rank) {
			fp1 = fopen(fname1, "rb");
			if (NULL == fp1 && 0 == rank) {
				fprintf(stderr, "Could not open file %s for reading, exiting\n", fname1);
				exit(1234);
			}
			#if (BHNSQ2)
			#if (BHNSQ2_1)
			ext.nx = 384;
			ext.ny = 96;
			ext.nz = 96;
			#else
			ext.nx = 384;
			ext.ny = 96;
			ext.nz = 96;
			#endif
			read_last_line(last_line, MAXLEN, fp1);
			sscanf(last_line, "%lf %lf %lf %*lf %*lf %*lf %*lf %*lf %*lf %*lf %*lf", &ext.xmax, &ext.ymax, &ext.zmax);
			rewind(fp1);

			read_first_line(first_line, MAXLEN, fp1);
			sscanf(first_line, "%lf %lf %lf %*lf %*lf %*lf %*lf %*lf %*lf %*lf %*lf", &ext.xmin, &ext.ymin, &ext.zmin);
			#if (!BHNSQ2_1)
			ext.xmin /= r_unit;
			ext.xmax /= r_unit;
			#endif
			if (0 == rank) {
				fprintf(stderr, "[%d] reading IC block: resolution (%dx%dx%dx%d), extent (%g,%g)x(%g,%g)x(%g,%g), files %s...", rank, ext.nvars, ext.nx, ext.ny, ext.nz, ext.xmin, ext.xmax, ext.ymin, ext.ymax, ext.zmin, ext.zmax, fname1);
				fflush(stderr);
			}

			#if (BHNSQ2_1)
			nx = 384;
			ny = 96;
			nz = 96;
			#else
			nx = 384;
			ny = 96;
			nz = 96;
			#endif
			nvars = ext.nvars;
			nitems = (size_t)nvars * nx * ny * nz;
			memsize = sizeof(double) * nitems;
			icdata = malloc(memsize);

			if (NULL == icdata) {
				fprintf(stderr, "[%5d] could not allocate memory of size %ld\n", rank, memsize);
				fclose(fp1);
				#if (!BHNSQ2)
				fclose(fp2);
				#endif
				exit(1235);
			}
			//read in the data block from file
			#else
			fp2 = fopen(fname2, "rb");
			if (NULL == fp2 && 0 == rank) {
				fprintf(stderr, "Could not open file %s for reading, exiting\n", fname2);
				fclose(fp1);
				exit(1234);
			}
			read_last_line(last_line, MAXLEN, fp1);
			sscanf(last_line, "%d %d %d %lf %lf %lf ", &ext.nx, &ext.ny, &ext.nz, &ext.xmax, &ext.ymax, &ext.zmax);

			//rewind the file to the beginning for subsequent reading
			rewind(fp1);

			//skip comment lines in the first file and read in the first non-comment line
			read_first_line(first_line, MAXLEN, fp1);
			sscanf(first_line, "%*d %*d %*d %lf %lf %lf ", &ext.xmin, &ext.ymin, &ext.zmin);
			//read_first_line leaves file at the start of the first non-comment line

			//skip comment lines in the second file
			read_first_line(first_line, MAXLEN, fp2);
			//read_first_line leaves file at the start of the first non-comment line

			//account for coordinates counted off from zero
			ext.nx += 1;
			ext.ny += 1;
			ext.nz += 1;
			
			ext.xmin/=r_unit;
			ext.xmax/=r_unit;

			if (0 == rank) {
				fprintf(stderr, "[%d] reading IC block: resolution (%dx%dx%dx%d), extent (%g,%g)x(%g,%g)x(%g,%g), files %s and %s...", rank, ext.nvars, ext.nx, ext.ny, ext.nz, ext.xmin, ext.xmax, ext.ymin, ext.ymax, ext.zmin, ext.zmax, fname1, fname2);
				fflush(stderr);
			}
			ext.nx = ext.nx / stride1;
			ext.ny = ext.ny / stride2;
			ext.nz = ext.nz / stride3;

			nx = ext.nx;
			ny = ext.ny;
			nz = ext.nz;
			nvars = ext.nvars;
			nitems = (size_t) nvars * nx * ny * nz;
			memsize = sizeof(double) * nitems;
			icdata = malloc(memsize);

			if(NULL == icdata) {
				fprintf(stderr,"[%5d] could not allocate memory of size %ld\n", rank, memsize);
				fclose(fp1);
				#if (!BHNSQ2)
				fclose(fp2);
				#endif
				exit(1235);
			}
			//read in the data block from file
			#endif

			#if (BHNSQ2)
			for (ii = 0; ii < nx; ii++) for (jj = 0; jj < ny; jj++) for (kk = 0; kk < nz; kk++) {
				//fprintf(stderr, "[%d] blah %d %d %d\n", rank, ii, jj, kk);

				//first file, containing grid and data information
				ptr1 = fgets(buf1, MAXLEN, fp1);
				if (NULL == ptr1) break;

				dd(ii, jj, kk, VARI) = (double)ii;
				dd(ii, jj, kk, VARJ) = (double)jj;
				dd(ii, jj, kk, VARK) = (double)kk;

				nitems_read = sscanf(ptr1, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", &dd(ii, jj, kk, VARR), &dd(ii, jj, kk, VARTHETA), &dd(ii, jj, kk, VARPHI), &dd(ii, jj, kk, VARRHO), &dd(ii, jj, kk, VARP), &dd(ii, jj, kk, VARYE), &dd(ii, jj, kk, VARMUDT), &dd(ii, jj, kk, VARUDPHI), &dd(ii, jj, kk, VARVUR), &dd(ii, jj, kk, VARVUTHETA), &dd(ii, jj, kk, VARVUPHI));
				#if (!BHNSQ2_1)
				dd(ii, jj, kk, VARR) /= r_unit;
				#endif
				//dd(ii, jj, kk, VARUDPHI) /= r_unit;

				nitems_expected = 11;
				if (nitems_expected != nitems_read) break;
			} 

			if (nitems_expected != nitems_read || ferror(fp1) || (NULL == ptr1 && !feof(fp1))) {
				fprintf(stderr, "[%5d] Error reading from file(s)\n", rank);
			}
			fclose(fp1); fp1 = NULL;
			#else 
			do {
				//first file, containing grid information
				ptr1 = fgets(buf1, MAXLEN, fp1);
				if(NULL == ptr1) break;

				//second file, containing data information
				ptr2 = fgets(buf2, MAXLEN, fp2);
				if (NULL == ptr2) break;

				nitems_read = sscanf(ptr1, "%d %d %d ", &ii, &jj, &kk);
				nitems_expected = 3;
				if(nitems_expected != nitems_read) break;

				if (ii % stride1 != 0 || jj % stride2 != 0 || kk % stride3 != 0) continue;
        
				ii = ii / stride1;
				jj = jj / stride2;
				kk = kk / stride3;

				dd(ii, jj, kk, VARI) = (double) ii;
				dd(ii, jj, kk, VARJ) = (double) jj;
				dd(ii, jj, kk, VARK) = (double) kk;

				nitems_read = sscanf(ptr1, "%*d %*d %*d %lf %lf %lf \n", &dd(ii, jj, kk, VARR), &dd(ii, jj, kk, VARTHETA), &dd(ii, jj, kk, VARPHI));
				dd(ii, jj, kk, VARR) /= r_unit;
				nitems_expected = 3;
				if(nitems_expected != nitems_read) break;

				nitems_read = sscanf(ptr2, "%lf %lf %lf %lf %lf %lf %lf %lf \n", &dd(ii,jj,kk,VARRHO), &dd(ii,jj,kk,VARP), &dd(ii,jj,kk,VARYE), &dd(ii,jj,kk,VARMUDT), &dd(ii,jj,kk,VARUDPHI), &dd(ii,jj,kk,VARVUR), &dd(ii,jj,kk,VARVUTHETA), &dd(ii,jj,kk,VARVUPHI));
				dd(ii, jj, kk, VARUDPHI) /= r_unit;
				nitems_expected = 8;
				if(nitems_expected != nitems_read) break;

			} while(!ferror(fp1) && !ferror(fp2) && NULL != ptr1 && NULL != ptr2);

			if( nitems_expected != nitems_read || ferror(fp1) || ferror(fp2) || (NULL == ptr1 && !feof(fp1)) || (NULL == ptr2 && !feof(fp2)) ) {
				fprintf(stderr,"[%5d] Error reading from file(s)\n", rank);
			}
			fclose(fp1); fp1 = NULL;
			fclose(fp2); fp2 = NULL;
			#endif

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

	rhomax = 0.;
	umax = 0.;

	#if(!NSY)
	tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif

	eccentricity = 0.0;
	double Tnu;
	double ucon[NDIM], utcon[NDIM];
	for (n = 0; n < n_active; n++){
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z,k, Tnu, res) firstprivate(r,th,phi,sth,cth, X, tilt, pos_new, udphi, mudt, prim)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			pos_new[1] = r;
			pos_new[2] = th;
			pos_new[3] = phi;

			sth = sin(th) ;
			cth = cos(th) ;
						
			/*prim[RHO] = dd(i, j, z, VARRHO);
			prim[UU] = dd(i, j, z, VARP) / (gam - 1);
			prim[U1] = dd(i, j, z, VARVUR);
			prim[U2] = dd(i, j, z, VARVUTHETA);
			prim[U3] = dd(i, j, z, VARVUPHI);
			#if (DO_YE)
			prim[YE] = dd(i, j, z, VARYE);
			#endif
			udphi = dd(i, j, z, VARUDPHI) / r_unit;
			*/
			res = interpolate_spec_prims(r, th, phi, ext, icdata, prim, &udphi, &mudt);
			if (res || prim[RHO] < 1e-13) {
				prim[RHO] = 1e-7 * RHOMIN;
				prim[UU] = 1e-7 * UUMIN;
				prim[U1] = 0.0;
				prim[U2] = 0.0;
				prim[U3] = 0.0;

				#if (DO_YE)
				prim[YE] = 1.0;
				#endif
				#if (DONUCLEAR)
				prim[XALPHA] = 0.0;
				prim[XATM] = 1.0;
				#endif
			}
			else {
				/* convert from BL 4-vel to relative 4-vel in internal (KS prime) coords */
				utilde_to_ucon(prim, udphi, mudt, n_ord[n], i, j, z);
				#if (DONUCLEAR)
				prim[XALPHA] = 0.0;
				prim[XATM] = 0.0;
				#endif
			}

			prim[B1] = 0.;
			prim[B2] = 0.;
			prim[B3] = 0.;

			// initialize neutrinos
			#if (NEUTRINOS_M1)
			for (int sp = 0; sp < NU_SPECIES; sp++) {
				prim[index_nu(UU_NU, sp)] = 1e-30;
				prim[index_nu(U1_NU, sp)] = prim[U1];
				prim[index_nu(U2_NU, sp)] = prim[U2];
				prim[index_nu(U3_NU, sp)] = prim[U3];

				Tnu = pow(prim[index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
				prim[index_nu(NUMBER_NU, sp)] = prim[index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tnu);
				//prim[index_nu(NUMBER_NU, sp)] = 1e-30;
			}
			#endif

			//copy back to full prim array
			PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k] = prim[k];

			if(prim[RHO] > rhomax) {
				#pragma omp critical
				rhomax = prim[RHO];
			}
		}
	}

	if (icdata) {
		free(icdata);
		icdata = NULL;
	}

	#if (MPI_enable)
	/*Share rhomax among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#endif

	/* Normalize the densities so that max(rho) = 1 */
	if (rank == 0){
		fprintf(stderr, "rhomax: %g\n", rhomax);
	}

	for (n = 0; n < n_active; n++) {
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] /= rhomax;
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] /= rhomax;
		}
	}
	rhomax = 1.;

	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);
	
	bound_prim(p, 1);

	set_mag();

	sourceflag=0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif

	#if DOHELM
	// Using density and pressure = (gam - 1) * u, find new u, using Helmholtz EOS
	for (n = 0; n < n_active; n++) {
		//#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] *= (gam - 1.);
			eos_mode_rhopres_u(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}

	// Apply the floors
	for (n = 0; n < n_active; n++) fixup(p, n_ord[n]);

	#if (DOHELM_TEMPERATURE)
	// Set temperatures given u:
	for (n = 0; n < n_active; n++) {
		//#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			eos_mode_rhou_temp_init(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO], &p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU], 
				#if (DO_YE)
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE], 
				#else 
				1.0,
				#endif
				p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]
				#if (DONUCLEAR)
				, p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XALPHA], p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][XATM]
				#endif
			);
		}
	}

	#if (NEUTRINOS_M1)
	for (n = 0; n < n_active; n++) {
	//#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z)
		ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][YE] < 1.0)
				init_neutrinos(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
		}
	}
	#endif
	#endif

	bound_prim(p, 1);
	#endif
}

//returns the pointer to the first non-comment line in the file fp
//size is the size of the s array
char* read_first_line(char *s, size_t size, FILE *fp)
{
  char *last_newline, *last_line, *ptr;
  size_t len;
  fpos_t pos;
  int is_success;

  /* rewind the file to the beginning */
  fseek(fp, 0L, SEEK_SET);
  do {
    is_success = !fgetpos(fp, &pos);
    ptr = fgets(s, size, fp);
  }
  while( NULL != ptr && '#' == ptr[0] );
  //restore the file position to the beginning of the line
  if( is_success && NULL != ptr ) {
    fsetpos(fp, &pos);
  }
  return(ptr);
}

//returns the pointer to the last line in the file fp
//size is the size of the s array
char* read_last_line(char *s, size_t size, FILE *fp)
{
  char *last_newline, *last_line;

  //subtract one to get the max number of characters in the string
  //(i.e., not counting the terminating '\0')
  size--;

  /* now read that many bytes from the end of the file */
  fseek(fp, -size, SEEK_END);
  size_t len = fread(s, sizeof(char), size, fp);

  /* don't forget the null terminator */
  s[len] = '\0';

  /* and find the last newline character (there must be one, right?) */
  last_newline = strrchr(s, '\n');
  //no newline within max_len bytes of file end
  if(NULL == last_newline) {
    return(NULL);
  }
  last_line = last_newline+1;

  if((int) strlen(last_line) == 0) {
    *last_newline = '\0';
    /* and find the last newline character (there must be one, right?) */
    last_newline = strrchr(s, '\n');
    //no newline within max_len bytes of file end
    if(NULL == last_newline) {
      return(NULL);
    }
    last_line = last_newline+1;
  }

  //the length of the last line
  len = len-(last_line-s);

  //move the last line to the beginning of s
  memmove(s, last_line, (len+1)*sizeof(char));

  return(s);
}

//define compact form for array indexing
#define d(ii,jj,kk) icdata[((ivar*nx+ii)*ny+jj)*nz+kk]

int interpolate_spec_var(double r, double th, double ph, extent ext, double* icdata, int ivar, double* val)
{
	double x, y, z, dx, dy, dz;
	double i, j, k, di, dj, dk;
	int i0, j0, k0, i1, j1, k1, nx, ny, nz;
	double c00, c01, c10, c11, c0, c1, c;
	int ii, jj, kk;
	double th0, th1;

	//limit th, ph to [0,pi], [0,2pi)
	if (th < 0) th = 0;
	if (th > M_PI) th = M_PI;
	if (ph >= 2. * M_PI) ph -= 2. * M_PI;
	if (ph < 0) ph += 2. * M_PI;

	nx = ext.nx;
	ny = ext.ny;
	nz = ext.nz;

	// Find the index in R such that r > R
	for (i0 = j0 = k0 = 0; i0 < nx; i0++) {
		if (dd(i0, j0, k0, VARR) > r) break;
	}
	i0--;
	if (i0 < 0 || i0 >= nx - 1) return(1);

	di = log2(r / dd(i0, j0, k0, VARR)) / log2(dd(i0 + 1, j0, k0, VARR) / dd(i0, j0, k0, VARR));
	i = i0 + di;

	// Find the index in TH such that th > TH 
	for (j0 = 0; j0 < ny; j0++) {
		th1 = dd(i0, j0, k0, VARTHETA) * (1 - di) + dd(i0 + 1, j0, k0, VARTHETA) * di;
		if (th1 > th) break;
	}
	j0--;
	if (j0 < 0) {
		j0 = 0;
		dj = 0;
	}
	else if (j0 >= ext.ny - 1) {
		j0 = ny - 1;
		dj = 0;
	}
	else {
		th0 = dd(i0, j0, k0, VARTHETA) * (1 - di) + dd(i0 + 1, j0, k0, VARTHETA) * di;
		dj = (th - th0) / (th1 - th0);
	}
	j = j0 + dj;

	// Index in phi
	dz = (ext.zmax - ext.zmin) / (nz - 1);
	k = (ph - ext.zmin) / dz;// -0.5;

	i1 = (int)ceil(i);
	j1 = (int)ceil(j);
	k0 = floor(k);
	k1 = (int)ceil(k);
	if (i0 < 0 || i1 >= nx || j0 < 0 || j1 >= ny || k0 < -1 || k1 >= nz + 1) {
		return(1);
	}
	dk = k - floor(k);
	if (k0 == -1) k0 = nz - 1;
	if (k1 == nz) k1 = 0;

	c =	d(i0, j0, k0) * (1. - di) * (1. - dj) * (1. - dk) +
		d(i0, j0, k1) * (1. - di) * (1. - dj) * (dk) +
		d(i0, j1, k0) * (1. - di) * (dj) * (1. - dk) +
		d(i0, j1, k1) * (1. - di) * (dj) * (dk) +
		d(i1, j0, k0) * (di) * (1. - dj) * (1. - dk) +
		d(i1, j0, k1) * (di) * (1. - dj) * (dk) +
		d(i1, j1, k0) * (di) * (dj) * (1. - dk) +
		d(i1, j1, k1) * (di) * (dj) * (dk);

	/*
	c00 = d(i0, j0, k0) * (1 - di) + d(i1, j0, k0) * di;
	c01 = d(i0, j0, k1) * (1 - di) + d(i1, j0, k1) * di;
	c10 = d(i0, j1, k0) * (1 - di) + d(i1, j1, k0) * di;
	c11 = d(i0, j1, k1) * (1 - di) + d(i1, j1, k1) * di;
	c0 = c00 * (1 - dj) + c10 * dj;
	c1 = c01 * (1 - dj) + c11 * dj;
	c = c0 * (1 - dk) + c1 * dk;
	*/
	if (isnan(c))  return(1);
	*val = c;
	return(0);
}

int interpolate_spec_prims(double r, double th, double ph, extent ext, double* data, double* p, double* udphi, double* mudt)
{
	int interpolate_spec_var(double r, double th, double ph, extent ext, double* data, int ivar, double* val);
	double vx, vy, vz, poten, x, y, z, R;
	double bl_gcov[NDIM][NDIM];
	int res;

	//vars: VARI, VARJ, VARK, VARR, VARTHETA, VARPHI, VARRHO, VARP, VARYE, VARMUDT, VARVUR, VARVUTHETA, VARVUPHI
	//ivar:  0,    1,    2,     3,    4,         5,     6,      7,    8,      9,      10,        11,       12
	res = interpolate_spec_var(r, th, ph, ext, data, VARRHO, &p[RHO]);
	//note that this is pressure, not internal energy
	res += interpolate_spec_var(r, th, ph, ext, data, VARP, &p[UU]); p[UU] /= (gam - 1);
	#if(DO_YE)
	res += interpolate_spec_var(r, th, ph, ext, data, VARYE, &p[YE]);
	#endif
	res += interpolate_spec_var(r, th, ph, ext, data, VARUDPHI, udphi);
	res += interpolate_spec_var(r, th, ph, ext, data, VARMUDT, mudt);
	res += interpolate_spec_var(r, th, ph, ext, data, VARVUR, &p[U1]);
	res += interpolate_spec_var(r, th, ph, ext, data, VARVUTHETA, &p[U2]);
	res += interpolate_spec_var(r, th, ph, ext, data, VARVUPHI, &p[U3]);

	p[B1] = 0.;
	p[B2] = 0.;
	p[B3] = 0.;

	return(res);
}