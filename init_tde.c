#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

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
      icdata = (double*)malloc(memsize);
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
	x = r * sin(th)*cos(ph);
	y = r * sin(th)*sin(ph);
	z = r * cos(th);
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