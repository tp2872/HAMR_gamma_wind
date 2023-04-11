#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

void init_torus_spherical()
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
	
	/* for Bondi */
	double n_adi,t_crit,rad_bondi;
	rad_bondi = 100.0;
	n_adi=1.0/(gam-1.0);
        t_crit=n_adi/(n_adi+1.0) / (2.0*rad_bondi) / (1.0-(n_adi+3.0)/(2.0*rad_bondi));
	/* disk parameters (use fishbone.m to select new solutions) */
	double temp = a;
	a = 0.95;
	rin = 20.0;
	rmax = 41.;
        l = lfish_calc(rmax) ;
	kappa = 1.e-3 ;
	beta = 1. ;
        rhomax = 1.0;
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

	//rhomax = pow(t_crit, n_adi);//1.0;
	//umax = kappa*pow(rhomax,gam)/(gam - 1.);//t_crit*pow(t_crit, n_adi)*gam/(gam-1.0)+pow(t_crit, n_adi);
	#if(!NSY)
	tilt = (TILT_ANGLE)/180.*M_PI;
	#else
	tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif
	eccentricity = 0.0;
	for (n = 0; n < n_active; n++){
		#pragma omp parallel for collapse(3) schedule(static,(BS_1*BS_2*BS_3)/nthreads) private(i,j,z) firstprivate(r,th,phi,sth,cth, ur,uh,up,u,rho,bl_gcov,X, X_cart, V, V_old, V_new, pos_new,tilt, eccentricity,geom, l,rin,lnh,expm2chi,up1, DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin,kappa,hm1,inmsg, rho_av,beta,bsq_ij,bsq_max,norm,q,beta_act,temp)
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

/*			DD = r*r - 2.*r + a*a ;
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

*/			
			rho = pow(t_crit, n_adi);//1.e-7*RHOMIN ;
			if (r < 6.0) {
                          rho = rho*exp(5.0*(1-6.0/r));
                        }
			u = t_crit*pow(t_crit, n_adi)/(gam-1.0);//t_crit*pow(t_crit, n_adi)*gam/(gam-1.0)+pow(t_crit, n_adi);//1.0;//1.e-7*UUMIN ;
			//if (r < 6.0){
			//	rho = 1.e-7*RHOMIN ;
                        //	u = 1.e-7*UUMIN ;
			//}
			ur = 0. ;
			uh = 0. ;
			up = 0. ;

			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][RHO] = rho;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][UU] = u;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U1] = ur;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U2] = uh;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][U3] = up;
			/* region inside magnetized torus; u^i is calculated in
			 * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
			 * so it needs to be transformed at the end */
			
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B1] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i,j,z)][B2] = 0.;
			p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][B3] = 0.;
			#if(RAD_M1)
			init_rad_pres(p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)]);
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
	//ZSLOOP(0,N1-1,0,N2-1) {
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] - 1, N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 - 1, N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 - 1) {
	//		p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][RHO] /= rhomax;
	//		p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][UU] /= rhomax;
	//	}
	//}
	//umax /= rhomax ;
	//rhomax = 1. ;
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}

	bound_prim(p, 1);

	set_mag_spherical();
                #if(CONSTANT_BC)
        int i1, k;
        for (n = 0; n < n_active; n++) {
                if (block[n_ord[n]][AMR_NBR2] == -1) {
                        for (j = N2_GPU_offset[n_ord[n]]; j < N2_GPU_offset[n_ord[n]] + BS_2; j++)for (z = N3_GPU_offset[n_ord[n]]; z < N3_GPU_offset[n_ord[n]] + BS_3; z++) {
                                /*
                                for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G; i++) {
                                        PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
                                        pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)];
                                        #if(STAGGERED)
                                        for (k = 2; k < NDIM; k++) {
                                                ps[nl[n]][index_3D(n, i, j, z)][k] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
                                        }
                                        #endif
                                        }*/
                                for (i = N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) + N1G; i++) {
                                        PLOOP p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k] = p[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][k];
                                        PLOOP ph[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][k] = p[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][k];

                                        ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][2];
                                        ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][3];
                                        psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][2];
                                        psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) - 1, j, z)][3];
                                }
                                for (i = N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1])+1; i < N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]) + N1G; i++) {
                                        ps[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]), j, z)][2];
                                        psh[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = ps[nl[n_ord[n]]][index_3D(n_ord[n], N1 * pow(1 + REF_1, block[n_ord[n]][AMR_LEVEL1]), j, z)][2];
                                }
                        }
                }
        }
        #endif
	bound_prim(p, 1);
	sourceflag=0.;
	#if(ELLIPTICAL2)
	calc_source();
	#endif
}


void set_mag_spherical(void){
	int i, j, z, k, n;
	double rhomax = 0., umax = 0.0;//0.007/(gam-1.0);
	int i100 = 0;
	double rho_av, q, beta = 1.0, bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM],pos_new[NDIM], beta_ij;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	#if(!NSY)
	double tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
	double tilt = -(TILT_ANGLE) / 180.*M_PI;
	#endif	
	
	double bin = 0.0000016785;
	double turb_coeff = 0.21;
	double coeff;

	static const size_t Nloops = 0;//10000;
		FILE* fp = fopen("3dloops_size.dat", "r+");
		int kk;
        double xc[Nloops], yc[Nloops], zc[Nloops], size[Nloops];
        double xx, yy, zz, dist;
        for (kk = 0;kk<Nloops;kk++){
                fscanf(fp, "%lf %lf %lf %lf", &xc[kk], &yc[kk], &zc[kk], &size[kk]);
        }
        //
        fclose(fp);

	do{
		i100++;
		coord(0, i100, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
	} while (r < 800.0);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]]-N1G, BS_1 + N1_GPU_offset[n_ord[n]] + D1, N2_GPU_offset[n_ord[n]] - N2G, N2_GPU_offset[n_ord[n]] + BS_2 + D2, N3_GPU_offset[n_ord[n]] - N3G, N3_GPU_offset[n_ord[n]] + BS_3+D3){
			dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][0] = 0.;
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

	/* first find corner-centered vector potential */
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(N1_GPU_offset[n_ord[n]]-N1G, BS_1 + N1_GPU_offset[n_ord[n]]+D1, N2_GPU_offset[n_ord[n]]-N2G, N2_GPU_offset[n_ord[n]] + BS_2+D2, N3_GPU_offset[n_ord[n]]-N3G, N3_GPU_offset[n_ord[n]] + BS_3+D3){
			/* Cell centered vector potential */	
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
		
			dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.5*bin*pow(r*sin(th),2.0); //code comparison
			if (r < 6.0) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.5*bin*pow(r*sin(th),2.0)*exp(5.0*(1-6.0/r));
			}
			xx = r*sin(th)*cos(phi);
			yy = r*cos(th)*sin(phi);
			zz = r*cos(th);
			for (kk=0;kk<Nloops;kk++){
				coeff = turb_coeff;
    				if (kk%2 == 0)
      					coeff = -turb_coeff;
				dist = size[kk] - pow(pow(xx - xc[kk], 2.0) + pow(yy - yc[kk], 2.0), 0.5);
				if (dist > 0.0){
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] += bin*coeff * r * sin(th) * dist * exp(-pow(size[kk]/dist, 2.0));
				}
			}

			#if (TILTED)
            V[1] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1];
            V[2] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][2];
            V[3] = dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3];
            rotate_vector2(V, pos_new, &r, &th, &phi, tilt);
            dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][1] = V[1];
            dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = V[2];
            dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = V[3];

            if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1])) {
                    dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][1] = 0.0;
                    //fprintf(stderr, "Error 1: (%d %d %d) r: %f th: %f phi: %f  r2: %f th2: %f phi2: %f \n", i, j, z, r, th, phi, pos_new[1], pos_new[2], pos_new[3]);
            }
            if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2])) {
                    dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
                    //fprintf(stderr, "Error 2: (%d %d %d) r: %f th: %f phi: %f  r2: %f th2: %f phi2: %f \n", i, j, z, r, th, phi, pos_new[1], pos_new[2], pos_new[3]);
            }
            if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3])) {
                    dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
                    //fprintf(stderr, "Error 3: (%d %d %d) r: %f th: %f phi: %f  r2: %f th2: %f phi2: %f \n", i, j, z, r, th, phi, pos_new[1], pos_new[2], pos_new[3]);
            }
			#endif

			#if(SPHERICAL || SPHERICAL_GR)
			if (j < 0 || j >= N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) {
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] *= -1.0;
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] *= -1.0;
			}
			#endif
			#if(CARTESIAN_GR)
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
	nstep = AMR_SWITCHTIMELEVEL - 1;
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
				E_corn[nl[n_ord[n]]][index_3D(n_ord[n], i, N2_GPU_offset[n_ord[n]], z)][1] = 0.;
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

	double bsq_max = 0.;
	double ug_sum = 0.;
	double bsq_sum = 0.;
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
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > umax && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)){
				umax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			if (bsq_ij > bsq_max && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				bsq_max = bsq_ij;
			}
			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r<50.) {
				bsq_sum += bsq_ij* geom.g* dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

	#if (MPI_enable)
	/*Share bsq_max among MPI processes*/
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	#endif
	#endif

	if (rank == 0){
		fprintf(stderr, "initial bsq_max: %g\n", bsq_max);
	}

	/* finally, normalize to set field strength */
	#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.)*ug_sum / (0.5*bsq_sum);
	#else
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	#endif
	if (rank == 0){
		fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	}
	norm = sqrt(beta_act / beta);

	/*for (n = 0; n < n_active; n++){
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
*/
	bsq_max = 0.;
	umax = 0;
	bsq_sum = 0.;
	ug_sum = 0.;
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);
			if (bsq_ij > bsq_max && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				bsq_max = bsq_ij;
			}
			if (p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > umax && (j > 4) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				umax = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}

			#if(WHICHPROBLEM==THIN_PROBLEM)
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005;
			coord(n_ord[n], i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			if (q > 0. && r<50.) {
				bsq_sum += bsq_ij* geom.g* dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
				ug_sum += p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] * geom.g * dx[nl[n_ord[n]]][1] * dx[nl[n_ord[n]]][2] * dx[nl[n_ord[n]]][3];
			}
			#endif
		}
	}

	/*Share bsq_max among MPI processes*/
	#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	#endif
	#endif

	#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.)*ug_sum / (0.5*bsq_sum);
	#else
	beta_act = (gam - 1.)*umax / (0.5*bsq_max);
	#endif
	if (rank == 0){
		fprintf(stderr, "final beta: %g (should be %g)\n", beta_act, beta);
	}

	/* enforce boundary conditions */
	for (n = 0; n < n_active; n++){
		fixup(p, n_ord[n]);
	}
	bound_prim(p, 1);
}

