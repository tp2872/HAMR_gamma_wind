/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/

/**
 *
 * this contains the generic piece of code for advancing
 * the primitive variables 
 *
**/

#include "decs.h"
/** algorithmic choices **/

/* use local lax-friedrichs or HLL flux:  these are relative weights on each numerical flux */
#define HLLF  (1.0)
#define LAXF  (0.0)

/** end algorithmic choices **/
double advance(double (*pi)[NPR], double (*pb)[NPR], double Dt, double (*pf)[NPR]);
double advance_GPU(double Dt, int flag);
double fluxcalc(double (*pr)[NPR], double (*F)[NPR], int dir);
double fluxcalc_GPU(int dir, int flag);
void   flux_ct(double(*F1)[NPR], double(*F2)[NPR], double(*F3)[NPR]);

/***********************************************************************************************/
/***********************************************************************************************
  step_ch():
  ---------
     -- handles the sequence of making the time step, the fixup of unphysical values, 
        and the setting of boundary conditions;

     -- also sets the dynamically changing time step size;

***********************************************************************************************/
void step_ch()
{
	double ndt, inmsg;
	int i, j, k;

	if (rank == 0){
		fprintf(stderr, "h");
	}
	ndt = advance(p, p, 0.5*dt, ph);
	fixup(ph);
	bound_prim(ph,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//bound_prim(ph, 1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	if (pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] == 100){
		fixup_utoprim(ph);  /* Fix the failure points using interpolation and updated ghost zone values */
		bound_prim(ph,0);    /* Reset boundary conditions with fixed up points */
		pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] = 0;
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	if (rank == 0){
		fprintf(stderr, "f");
	}

	ndt = advance(p, ph, dt, p);
	fixup(p);
	bound_prim(p,1);
	//bound_prim(p, 1);
	if (pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] == 100){
		fixup_utoprim(p);
		bound_prim(p,0);
		pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] = 0;
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&dt, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, Stat);
			dt = MY_MIN(dt, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&dt, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}

	if (rank != 0){
		rc = MPI_Recv(&dt, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */

	/* done! */
}

/***********************************************************************************************/
/***********************************************************************************************
advance():
---------
-- responsible for what happens during a time step update, including the flux calculation,
the constrained transport calculation (aka flux_ct()), the finite difference
form of the time integral, and the calculation of the primitive variables from the
update conserved variables;
-- also handles the "fix_flux()" call that sets the boundary condition on the fluxes;

***********************************************************************************************/
double advance(
	double(*pi)[NPR],
	double(*pb)[NPR],
	double Dt,
	double(*pf)[NPR]
	)
{
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U[NPR], dU[NPR];
	struct of_geom geom;
	struct of_state q;
	double U_ent;
	#pragma omp parallel shared(pf,pi) private(i,j,z, k)
	{
		#pragma omp for schedule(dynamic,1)
		ZLOOP3D_MPI{
			//#pragma omp simd
			PLOOP pf[index(i, j, z)][k] = pi[index(i, j, z)][k];        /* needed for Utoprim */
		}
	}

	//fprintf(stderr, "0");
	#if(N1G>0)
	ndt1 = fluxcalc(pb, F1, 1);
	#else
	ndt1 = 1e9;
	#endif
	#if(N2G>0)
	ndt2 = fluxcalc(pb, F2, 2);
	#else
	ndt2 = 1e9;
	#endif
	#if(N3G>0)
	ndt3 = fluxcalc(pb, F3, 3);
	#else
	ndt3 = 1e9;
	#endif
	fix_flux(F1, F2, F3);
	flux_ct(F1, F2, F3);

	//scanf("%d", &l);
	/* evaluate diagnostics based on fluxes */
	//diag_flux(F1);

	//fprintf(stderr, "1");
	/** now update pi to pf **/
	#pragma omp parallel shared(pi,pb, pf, Katm, failimage, Dt, F1, F2,F3, pflag, dx) private(i,j,z,k, geom, q, U, dU)
	{
		#pragma omp for schedule(static,1)
		ZLOOP_MPI{
			get_geometry(i, j, CENT, &geom);
			for (z = N3_MPI_offset; z < N3_MPI_offset + N3_MPI; z++){
				source(pb[index(i, j, z)], &geom, i, j, dU, Dt);
				get_state(pi[index(i, j, z)], &geom, &q);
				primtoU(pi[index(i, j, z)], &q, &geom, U);
				//#pragma omp simd
				PLOOP{
					U[k] += Dt*(
					#if( N1G > 0 )
					- (F1[index(i + 1, j, z)][k] - F1[index(i, j, z)][k]) / dx[1]
					#endif
					#if( N2G > 0 )
					- (F2[index(i, j + 1, z)][k] - F2[index(i, j, z)][k]) / dx[2]
					#endif
					#if( N3G > 0 )
					- (F3[index(i, j, z + 1)][k] - F3[index(i, j, z)][k]) / dx[3]
					#endif
					+ dU[k]);
				}
				pflag[index(i, j, z)] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf[index(i, j, z)]);

				#if( DO_FONT_FIX ) 
				if (pflag[index(i, j, z)]) {
					failimage[index(i, j, z)][0]++;
					U_ent = (geom.g *pi[index(i, j, z)][0] * (gam - 1.)*pi[index(i, j, z)][1] / pow(pi[index(i, j, z)][0], gam)) * (q.ucon[0]);
					pflag[index(i, j, z)] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf[index(i, j, z)], U_ent);
					if (pflag[index(i, j, z)]) {
						failimage[index(i, j, z)][1]++;
						//pflag[index(i, j, z)] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf[index(i, j, z)], U_ent);
						if (pflag[index(i, j, z)]){
							pflag[index(-2, -2, -2)] = 100;
							failimage[index(i, j, z)][2]++;
						}
					}
				}
				#else
				if (pflag[index(i, j, z)]) {
					pflag[index(-2, -2, -2)] = 100;
					failimage[index(i, j, z)][0]++;
				}
				#endif
			}
		}
	}
	ndt = defcon * 1. / (1. / ndt1 + 1. / ndt2 + 1. / ndt3);
	//fprintf(stderr, "2");

	return(ndt);
}

/***********************************************************************************************/
/***********************************************************************************************
fluxcalc():
---------
-- sets the numerical fluxes, avaluated at the cell boundaries using the slope limiter
slope_lim();

-- only has HLL and Lax-Friedrichs  approximate Riemann solvers implemented;

***********************************************************************************************/
double fluxcalc(double(*pr)[NPR], double(*F)[NPR], int dir)
{
	int i, j, z, k, idel, jdel, zdel, face;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR];
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, ndt, ndt_thread, dtij;
	double ctop;
	struct of_geom geom;
	struct of_state state_l, state_r;
	double bsq;
	int max_i, max_j, max_z;
	double val;
	ndt = 1.e9;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }
	else { exit(10); }

	#pragma omp parallel shared(pr, dq, ndt, cour, dx,dir, F, face, idel, jdel, zdel) private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l, state_r, F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin,ctop, dtij)
	{
		ndt_thread = 1.e9;
		#if(RESCALE)
		/** evaluate slopes of primitive variables **/
		/* first rescale */
		#pragma omp for schedule(static,1)
		ZSLOOP(N1_MPI_offset - 2, N1_MPI_offset + N1_MPI + 1, N2_MPI_offset - 2, N2_MPI_offset + N2_MPI + 1) 	{
			get_geometry(i, j, CENT, &geom);
			rescale(pr[index(i, j, z)], FORWARD, dir, i, j, CENT, &geom);
		}
		#endif

		/* then evaluate slopes */
		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_MPI_offset - D1, N1_MPI_offset + N1_MPI - 1 + D1, N2_MPI_offset - D2, N2_MPI_offset + N2_MPI - 1 + D2, N3_MPI_offset - D3, N3_MPI_offset + N3_MPI - 1 + D3){
			//#pragma omp simd
			PLOOP{
				dq[index(i, j, z)][k] = slope_lim(pr[index(i - idel, j - jdel, z - zdel)][k], pr[index(i, j, z)][k], pr[index(i + idel, j + jdel, z + zdel)][k]);
			}
		}

		#pragma omp for schedule(static,1)
		ZSLOOP((N1_MPI_offset - jdel - zdel)*D1, (N1_MPI_offset + N1_MPI)*D1, (N2_MPI_offset - idel - zdel)*D2, (N2_MPI_offset + N2_MPI)*D2) 	{
			get_geometry(i, j, face, &geom);
			//if (i == 50 && j == 50)
			//{
			//	printf("testCPU: %f %d \n", geom.gcov[0][0],face);
			//}
			for (z = (N3_MPI_offset - idel - jdel)*D3; z <= (N3_MPI_offset + N3_MPI)*D3; z++){
				/* this avoids problems on the pole */
				if (dir == 2 && (j == 0 || j == N2)) {
					//#pragma omp simd
					PLOOP F[index(i, j, z)][k] = 0.;
				}
				else {
					//#pragma omp simd
					PLOOP{
						p_l[k] = pr[index(i - idel, j - jdel, z - zdel)][k] + 0.5*dq[index(i - idel, j - jdel, z - zdel)][k];
						p_r[k] = pr[index(i, j, z)][k] - 0.5*dq[index(i, j, z)][k];
					}

					#if(RESCALE)
					get_geometry(i, j, CENT, &geom);
					rescale(p_l, REVERSE, dir, i, j, face, &geom);
					rescale(p_r, REVERSE, dir, i, j, face, &geom);
					#endif

					get_state(p_l, &geom, &state_l);
					get_state(p_r, &geom, &state_r);

					primtoflux(p_l, &state_l, dir, &geom, F_l);
					primtoflux(p_r, &state_r, dir, &geom, F_r);

					primtoflux(p_l, &state_l, TT, &geom, U_l);
					primtoflux(p_r, &state_r, TT, &geom, U_r);

					vchar(p_l, &state_l, &geom, dir, &cmax_l, &cmin_l);
					vchar(p_r, &state_r, &geom, dir, &cmax_r, &cmin_r);

					cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
					cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
					ctop = MY_MAX(cmax, cmin);

					//#pragma omp simd
					PLOOP F[index(i, j, z)][k] = HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
						+ LAXF*(0.5*(F_l[k] + F_r[k] - ctop*(U_r[k] - U_l[k])));

					/* evaluate restriction on timestep */
					cmax = MY_MAX(cmax, cmin);
					dtij = cour*dx[dir] / cmax;
					if (dtij < ndt_thread) {
						ndt_thread = dtij;
					}

				}
			}
		}

		#pragma omp critical
		{
			if (ndt_thread < ndt){
				ndt = ndt_thread;
			}
		}

		#if(RESCALE)
		#pragma omp for schedule(static,1)
		ZSLOOP(N1_MPI_offset - 2, N1_MPI_offset + N1_MPI + 1, N2_MPI_offset - 2, N2_MPI_offset + N2_MPI + 1) 	{
			get_geometry(i, j, CENT, &geom);
			rescale(pr[index(i, j, z)], REVERSE, dir, i, j, CENT, &geom);
		}
		#endif
	}
	return(ndt);
}

/***********************************************************************************************/
/***********************************************************************************************
flux_ct():
---------
-- performs the flux-averaging used to preserve the del.B = 0 constraint (see Toth 2000);
Note that we use in this new version of HARM dq instead of emf as temporary storage!

***********************************************************************************************/
void flux_ct(double(*F1)[NPR], double(*F2)[NPR], double(*F3)[NPR])
{
	int i, j, z;

	/* calculate EMFs */
	/* Toth approach: just average */
	#pragma omp parallel shared(dq, F1, F2, F3) private(i,j,z)
	{
		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1 + D1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1 + D2, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1 + D3){
			#if (N2G>0 && N3G>0)
			dq[index(i, j, z)][1] = 0.25*(F2[index(i, j, z)][B3] + F2[index(i, j, z - 1)][B3] - F3[index(i, j, z)][B2] - F3[index(i, j - 1, z)][B2]);
			#endif
			#if (N1G>0 && N3G>0)
			dq[index(i, j, z)][2] = 0.25*(F3[index(i, j, z)][B1] + F3[index(i - 1, j, z)][B1] - F1[index(i, j, z)][B3] - F1[index(i, j, z - 1)][B3]);
			#endif
			#if (N1G>0 && N2G>0)
			dq[index(i, j, z)][3] = 0.25*(F1[index(i, j, z)][B2] + F1[index(i, j - 1, z)][B2] - F2[index(i, j, z)][B1] - F2[index(i - 1, j, z)][B1]);
			#elif
			dq[index(i, j, z)][3] = 0.25*(F1[index(i, j, z)][B2] + F1[index(i, j - 1, z)][B2]);
			#endif
		}

		/* rewrite EMFs as fluxes, after Toth */
		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1 + D1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) 	{
			#if (N1G>0)
			F1[index(i, j, z)][B1] = 0.;
			#endif
			#if (N1G>0 && N2G>0)
			F1[index(i, j, z)][B2] = 0.5*(dq[index(i, j, z)][3] + dq[index(i, j + 1, z)][3]);
			#endif
			#if (N1G>0 && N3G>0)
			F1[index(i, j, z)][B3] = -0.5*(dq[index(i, j, z)][2] + dq[index(i, j, z + 1)][2]);
			#endif
		}

		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1 + D2, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) 	{
			#if (N1G>0 && N2G>0)		
			F2[index(i, j, z)][B1] = -0.5*(dq[index(i, j, z)][3] + dq[index(i + 1, j, z)][3]);
			#endif
			#if (N2G>0 && N3G>0)
			F2[index(i, j, z)][B3] = 0.5*(dq[index(i, j, z)][1] + dq[index(i, j, z + 1)][1]);
			#endif
			#if(N2>1)
			F2[index(i, j, z)][B2] = 0.;
			#endif
		}
		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1 + D3) 	{
			#if (N1G>0 && N3G>0)
			F3[index(i, j, z)][B1] = 0.5*(dq[index(i, j, z)][2] + dq[index(i + 1, j, z)][2]);
			#endif
			#if (N2G>0 && N3G>0)
			F3[index(i, j, z)][B2] = -0.5*(dq[index(i, j, z)][1] + dq[index(i, j + 1, z)][1]);
			#endif
			#if(N3G>0)
			F3[index(i, j, z)][B3] = 0.;
			#endif
		}
	}
}

void GPU_step_ch()
{
	double ndt, inmsg;
	int i;
	/*
	if (rank == 0){
		fprintf(stderr, "h");
	}
	*/
	ndt = advance_GPU(0.5*dt, 0);   /* time step primitive variables to the half step */
	GPU_fixup(0);
	GPU_boundprim(0 ,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	/* Repeat and rinse for the full time (aka corrector) step:  */
	/*
	if (rank == 0){
		fprintf(stderr, "f");
	}
	*/
	
	ndt = advance_GPU(dt, 1);   /* time step primitive variables to the half step */
	GPU_fixup(1);
	GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&dt, 1, MPI_DOUBLE, 0, 500, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 500, MPI_COMM_WORLD, Stat);
			dt = MY_MIN(dt, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&dt, 1, MPI_DOUBLE, i, 1000, MPI_COMM_WORLD, &reqs[i+numtasks]);
		}
	}

	if (rank != 0){
		rc = MPI_Recv(&dt, 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD, Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
	/* done! */
}

double advance_GPU(double Dt, int flag)
{
	double ndt,ndt1,ndt2,ndt3;
	#if(N1G>0)
	ndt1 = fluxcalc_GPU(1, flag);
	#else
	ndt1 = 1e9;
	#endif
	#if(N2G>0)
	ndt2 = fluxcalc_GPU(2, flag);
	#else
	ndt2 = 1e9;
	#endif
	#if(N3G>0)
	ndt3 = fluxcalc_GPU(3, flag);
	#else
	ndt3 = 1e9;
	#endif
	GPU_fix_flux();
	GPU_flux_ct1();
	GPU_flux_ct2();
	//GPU_diag_flux();
	GPU_Utoprim(flag, Dt);
	hslope = ndt3 / ndt2;
	ndt = defcon * 1. / (1. / ndt1 + 1. / ndt2+1./ndt3);
	return(ndt) ;
}



void GPU_finish(void)
{

}

/*This function determines the speed-up factor of the OpenCL version over the normal C++ version*/
void GPU_benchmark(void)
{
	int i;
	clock_t begin, end;
	double time_spent1, time_spent2;

	/*Time both the CPU and GPU versions of the code for a given number of steps to hide overhead*/
	begin = clock();
	for (i = 0; i < 15; i++){
		GPU_step_ch();
	}
	clFinish(commandQueueGPU);
	end = clock();
	time_spent1 = (double)(end - begin) / CLOCKS_PER_SEC;
	begin = clock();
	for (i = 0; i < 15; i++){
		step_ch();
	}
	end = clock();
	time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
	
	fprintf(stderr, "\nSpeedup : %f\n", time_spent2 / time_spent1);
}

/*Used for debugging. Compares output from CPU version to output from GPU version*/
void step_ch_debug()
{
	double ndt, inmsg;
	int i, j, z, k;
	MPI_Barrier(MPI_COMM_WORLD);
	ndt = advance_GPU(0.5*dt, 0);   /* time step primitive variables to the half step */
	//GPU_fixup(0);
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(0);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//clFinish(commandQueueGPU);
	fprintf(stderr, "\n h_dt(GPU%d): %f     ", rank, ndt);

	ndt = advance(p, p, 0.5*dt, ph);
	//fixup(ph);
	//bound_prim(ph,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//fixup_utoprim(ph);  /* Fix the failure points using interpolation and updated ghost zone values */
	//bound_prim(ph,1);    /* Reset boundary conditions with fixed up points */
	fprintf(stderr, "h_dt(CPU%d): %f \n ", rank, ndt);

	/*Temporary store ph array from CPU to test array so ph array from GPU can be loaded*/
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G-1) {
		PLOOP{
			F1[index(i, j, z)][k] = ph[index(i, j, z)][k];
			F2[index(i, j, z)][k] = p[index(i, j, z)][k];
		}
	}

	/*Read ph array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	code produce inconsistent output*/
	GPU_read();
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G - 1) {
		PLOOP{
			if (ph[index(i, j, z)][k] / F1[index(i, j, z)][k] > 1.001 || ph[index(i, j, z)][k] / F1[index(i, j, z)][k] < 0.999){
				fprintf(stderr, " i1:%d, j:%d, z:%d, k: %d, rank:% d, value1: %f value2: %f  \n", i, j, z, k, rank,
					log(ph[index(i, j, z)][k] * ph[index(i, j, z)][k]) / log(10.), log(F1[index(i, j, z)][k] * F1[index(i, j, z)][k]) / log(10.));
			}
		}
	}

	/*Restore CPU version of ph array*/
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G - 1) {
		PLOOP{
			ph[index(i, j, z)][k] = F1[index(i, j, z)][k];
			p[index(i, j, z)][k] = F2[index(i, j, z)][k];
		}
	}
	
	/* Repeat and rinse for the full time (aka corrector) step:  */
	MPI_Barrier(MPI_COMM_WORLD);
	ndt = advance_GPU(dt,1);   /* time step primitive variables to the half step */
	//GPU_fixup(1);
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(1);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//clFinish(commandQueueGPU);
	fprintf(stderr, "f_dt(GPU%d): %f     ", rank, ndt);
	
	ndt = advance(p, ph, dt, p);
	//fixup(p);
	//bound_prim(p,1);
	//fixup_utoprim(p);
	//bound_prim(p,1);
	fprintf(stderr, "f_dt(CPU%d): %f\n ", rank, ndt);
	
	/*Temporary store p array from CPU to test array so ph array from GPU can be loaded*/
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G - 1) {
		PLOOP{
			F1[index(i, j, z)][k] = p[index(i, j, z)][k];
			F2[index(i, j, z)][k] = ph[index(i, j, z)][k];
		}
	}

	/*Read p array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	code produce inconsistent output*/
	GPU_read();
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G - 1) {
		PLOOP{
			if (p[index(i, j, z)][k] / F1[index(i, j, z)][k]>1.001 || p[index(i, j, z)][k] / F1[index(i, j, z)][k] < 0.999){
				fprintf(stderr, " i2:%d, j:%d, z: %d, k: %d, rank: %d, value1: %f, value2: %f  \n", i, j, z, k, rank,
					log(p[index(i, j, z)][k] * p[index(i, j, z)][k]) / log(10.), log(F1[index(i, j, z)][k] * F1[index(i, j, z)][k]) / log(10.));
			}
		}
	}
		
	/*Restore CPU version of p array*/
	ZSLOOP3D(-2 + N1_MPI_offset, N1_MPI + N1_MPI_offset + 1, -2 + N2_MPI_offset, N2_MPI_offset + N2_MPI + 1, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI + N3G - 1) {
		PLOOP{
			p[index(i, j, z)][k] = F1[index(i, j, z)][k];
			ph[index(i, j, z)][k] = F2[index(i, j, z)][k];
		}
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank != 0){
		rc = MPI_Isend(&dt, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[rank]);
	}
	else{
		for (i = 1; i < numtasks; i++){
			rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, Stat);
			dt = MY_MIN(dt, inmsg);
		}
		for (i = 1; i < numtasks; i++){
			rc = MPI_Isend(&dt, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &reqs[i + numtasks]);
		}
	}

	if (rank != 0){
		rc = MPI_Recv(&dt, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, Stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
	
	/* done! */
}
