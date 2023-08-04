#include "include.h"
#include <float.h>
#include <complex.h>
#include "decs_MPI.h"

void set_mag(void){
	int i, j, z, k, n;
	double rhomax = 1., pmax = 0.;
	int i100 = 0;
	double rho_av, q, beta = BETA, bsq_ij, norm, beta_act, V[NDIM], X_cart[NDIM],pos_new[NDIM], beta_ij;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	struct of_state state;
	double gamma_g;

	#if(!NSY || CARTESIAN_GR)
	double tilt = (TILT_ANGLE) / 180.*M_PI;
	#else
	double tilt = -(TILT_ANGLE) / 180.*M_PI;
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

	do{
		i100++;
		coord(0, i100, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
	} while (r < 400.0);

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
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax-0.0005;
			#elif(WHICHPROBLEM==BONDI_PROBLEM_2D)
			if (r >= rin) q = (r * r - rin * rin) * (sin(th) * sin(th));
			else q = 0.;
			#elif(WHICHPROBLEM==COLLAPSAR || (WHICHPROBLEM==POSTMERGER_PROBLEM && FORNAX_IC) || WHICHPROBLEM==NSM)
				#if(COLLAPSAR_GR1D)
				q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax-0.0005;
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
			q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.0005; //Postmerger problem
			//q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax - 0.2; //SANE
			//q = p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][RHO] / rhomax*pow(r/20.*sin(th),3.)*exp(-r/400.) - 0.2; //code comparison
			#endif
			if (q > 0.){	
				#if(WHICHPROBLEM==THIN_PROBLEM)
				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = q*pow(r,2.0); //Toroidal
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = sin(2.0*M_PI *r/120.)*sqrt(r*r*r*r*r)*q;
				#else
				dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = q; //SANE+CODE_COMPARISON
				//dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = pow(q, 2.0) * pow(r, 3.0); //MAD
				#endif
			}
			else{
				dq[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)][3] = 0.0;
			}

			if (q > 0.) {
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
				}
				if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2])) {
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][2] = 0.0;
				}
				if (!isfinite(dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3])) {
					dq[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][3] = 0.0;
				}
				#endif
			}
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

			beta_ij = 2.0*(gam - 1.0)*p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] / bsq_ij;
			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0/3.0;
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
			if (((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 4) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 4)) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU]+ (4./3.-1.)*p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else

			if ((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && j > (int)(20.0/180.0* N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && j < (int)(N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))){
				pmax = (gamma_g-1.)*p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
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
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
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
	beta_act = pmax / (0.5*bsq_max);
	#endif
	if (rank == 0) fprintf(stderr, "initial beta: %g (should be %g)\n", beta_act, beta);
	norm = sqrt(beta_act / beta);

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
	for (n = 0; n < n_active; n++){
		ZLOOP3D_MPI{
			get_geometry(n_ord[n], i, j, z, CENT, &geom);
			bsq_ij = bsq_calc(p[nl[n_ord[n]]][index_3D(n_ord[n] ,i, j, z)], &geom);

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

			if (bsq_ij > bsq_max && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				bsq_max = bsq_ij;
			}

			#if(TWO_T)
				#if(RAD_M1 && HIGH_MDOT)
				gamma_g = 4.0/3.0;
				#else
				gamma_g = GAMMA;
				#endif
			#else
			gamma_g = GAMMA;
			#endif

			#if(RAD_M1)
			if (((gamma_g - 1.) *p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD]) > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] + (4. / 3. - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU_RAD];
			}
			#else
			if ((gamma_g - 1.) * p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU] > pmax && (j > 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2])) && (j < N2*pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]) - 20.0 / 180.0 * N2 * pow(1 + REF_2, block[n_ord[n]][AMR_LEVEL2]))) {
				pmax = (gamma_g-1.)*p[nl[n_ord[n]]][index_3D(n_ord[n], i, j, z)][UU];
			}
			#endif
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
	MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_DOUBLE, MPI_MAX, mpi_cartcomm);
	#if(WHICHPROBLEM==THIN_PROBLEM)
	MPI_Allreduce(MPI_IN_PLACE, &bsq_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &ug_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_cartcomm);
	#endif
	#endif

	#if(WHICHPROBLEM==THIN_PROBLEM)
	beta_act = (gam - 1.) * ug_sum / (0.5*bsq_sum);
	#else
	beta_act = pmax / (0.5*bsq_max);
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
