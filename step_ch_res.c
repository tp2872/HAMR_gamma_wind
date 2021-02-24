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

#include "decs_MPI.h"

void step_ch_res()
{
	double ndt, inmsg;
	int i, j, k, n, u;

	if (rank == 0){
		fprintf(stderr, "h");
	}

	for (u = 0; u < 2*AMR_MAXTIMELEVEL; u++){
		set_prestep();
		ndt = advance_res(0);

		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  fixup(p, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup(ph, n_ord[n]);
		}

		bound_prim(ph, 0);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
		nstep++;
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	if (rank == 0){
		fprintf(stderr, "f");
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += (double)(AMR_MAXTIMELEVEL)*dt;

	/*Calculate smallest timestep for all MPI threads*/
	#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &ndt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	#endif

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
    dt = ndt;

	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel(0);

	if ((t + dt * (double)AMR_SWITCHTIMELEVEL) > tf) dt = (tf - t) / ((double)AMR_SWITCHTIMELEVEL);  /* but don't step beyond end of run */
	/* done! */
}

double advance_res(int flag)
{
	int i, j, z, k, n, flag_local;
	double ndt, U[NPR], dU[NPR];
	int ind0, ind1, ind2, ind3;

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#pragma omp parallel shared(n,p, ph, n_ord, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset,nthreads) private(i,j,z,k)
			{
				#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
				ZLOOP3D_MPI{
					ind0 = index_3D(n_ord[n], i, j, z);
					#pragma ivdep
					PLOOP ph[nl[n_ord[n]]][ind0][k] = p[nl[n_ord[n]]][ind0][k];        /* needed for Utoprim */
				}
			}
		}
	}

	for (n = 0; n < n_active; n++){
		prestep_half[nl[n_ord[n]]] = (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) < block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
		prestep_full[nl[n_ord[n]]] = (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
		|| (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) < 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) >  block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
	}

	ndt1 = ndt2 = ndt3 = 1e9;
	for (n = 0; n < n_active; n++){
		bdt[nl[n_ord[n]]][0] = bdt[nl[n_ord[n]]][1] = bdt[nl[n_ord[n]]][2] = bdt[nl[n_ord[n]]][3] = 1e9;
	}

	#if(DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_0_res(dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
			fixup(p, n_ord[n]);
		}
	}
	#endif

	#if(N1G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][1] = fluxcalc_res(ph, F1, 1, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][1] = fluxcalc_res(p, F1, 1, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
				ndt1 = MY_MIN(ndt1, bdt[nl[n_ord[n]]][1]);
		}
		else{
			ndt1 = MY_MIN(ndt1, bdt[nl[n_ord[n]]][1] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}

	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send1(F1, Bufferp_1, n_ord[n]);
	set_iprobe(0, &flag_local);
	do{
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec1(F1, Bufferp_1, n_ord[n], 1);
		set_iprobe(1, &flag_local);
	} while (flag_local);
	set_iprobe(0, &flag_local);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec1(F1, Bufferp_1, n_ord[n], 2);

	#endif
	#if(N2G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[nl[n_ord[n]]][2] = fluxcalc_res(ph, F2, 2, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][2] = fluxcalc_res(p, F2, 2, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
			ndt2 = MY_MIN(ndt2, bdt[nl[n_ord[n]]][2]);
		}
		else{
			ndt2 = MY_MIN(ndt2, bdt[nl[n_ord[n]]][2] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send2(F2, Bufferp_1, n_ord[n]);
	do{
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec2(F2, Bufferp_1, n_ord[n], 1);
		set_iprobe(1, &flag_local);
	} while (flag_local);
	set_iprobe(0, &flag_local);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec2(F2, Bufferp_1, n_ord[n], 2);

	#endif
	#if(N3G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[nl[n_ord[n]]][3] = fluxcalc_res(ph, F3, 3, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][3] = fluxcalc_res(p, F3, 3, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
			ndt3 = MY_MIN(ndt3, bdt[nl[n_ord[n]]][3]);
		}
		else{
			ndt3 = MY_MIN(ndt3, bdt[nl[n_ord[n]]][3] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send3(F3, Bufferp_1, n_ord[n]);
	do{
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec3(F3, Bufferp_1, n_ord[n], 1);
		set_iprobe(1, &flag_local);
	} while (flag_local);
	set_iprobe(0, &flag_local);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec3(F3, Bufferp_1, n_ord[n], 2);
	#endif
	#if(!TRANS_BOUND && !CARTESIAN)
	for (n = 0; n < n_active; n++) if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fix_flux(F1, F2, F3, n_ord[n]);
	#endif
	#if(!STAGGERED)
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_ct(F1, F2, F3, n_ord[n]);
	#else
	#if(DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1_M1_2_res(ph, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1_res(p, n_ord[n]);
	}
	#else
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1_res(ph, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1_res(p, n_ord[n]);
	}
	#endif
	const_transport_bound();
	#if(DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2_res(ps, ps, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2_res(ps, psh, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#else
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  const_transport2_res(ps, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2_res(ps, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif
	#endif
	#if(DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_2_res(dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_1_res(dt* (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		}
	}
	#else
	for (n = 0; n < n_active; n++){
        if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  utoprim_res(p, ph, p, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) utoprim_res(p, p, ph, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif
	ndt = 1e9;

	for (n = 0; n < n_active; n++){
		bdt[nl[n_ord[n]]][0] = 1. / (1. / bdt[nl[n_ord[n]]][1] + 1. / bdt[nl[n_ord[n]]][2] + 1. / bdt[nl[n_ord[n]]][3]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
			ndt = MY_MIN(ndt, bdt[nl[n_ord[n]]][0]);
		}
		else{
			ndt = MY_MIN(ndt, bdt[nl[n_ord[n]]][0] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	
	//ndt=defcon*1./(1./ndt1+1./ndt2+1./ndt3);

	return defcon*ndt;
}

void utoprim_M1_0_res(double Dt, int n)
{
	#if(RESISTIVE)
	int i, j, z, k, ind0;
	double U_0[NPR];
	struct of_geom geom;
	struct of_state_res q_res;

	#pragma omp  parallel shared(n, p, Dt, pflag, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset) private(i, j, z, k, geom, U_0, q_res, ind0)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n]-N1G, N1_GPU_offset[n] + BS_1 + N1G - 1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G - 1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 +N3G - 1) {
			k = 1;

			if (k == 1) {
				ind0 = index_3D(n, i, j, z);
				get_geometry(n, i, j, z, CENT, &geom);
				get_state_res(p[nl[n]][ind0], &geom, &q_res);
				primtoflux_res(p[nl[n]][ind0], &q_res, 0, &geom, U_n[nl[n]][ind0]);

				//implicit_res_solve(p[nl[n]][ind0], U_n[nl[n]][ind0], U_n[nl[n]][ind0], U_0, &pflag[nl[n]][ind0], &pflag_rad[nl[n]][ind0], &geom, dU_RAD0[nl[n]][ind0], Dt * Y_IMEX, cell_size);
			}
		}
	}
	#endif
}

void utoprim_M1_1_res(double Dt, int n){
	#if(RESISTIVE)
	int i, j, z, k;
	double cell_size, U_1[NPR], q;
	struct of_geom geom;
	int ind0, ind1, ind2, ind3;

	#pragma omp  parallel shared(n, gdet, psh, dU_MHD1, Dt, F1, F2, F3, dx, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset, nthreads, gam) private(i, j, z, k, U_1, geom, ind0, ind1, ind2, ind3, q)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			get_geometry(n, i, j, z, CENT, &geom);

			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);

			divE_calc(p,n,i,j,z);
			source_res(p[nl[n]][ind0], &geom, n, i, j, z, dU_MHD1[nl[n]][ind0], &q, Dt);

			#pragma ivdep
			PLOOP{
				U_1[k] =  U_n[nl[n]][ind0][k] + Dt * (
				#if( N1G > 0 )
				- (F1[nl[n]][ind1][k] - F1[nl[n]][ind0][k]) / dx[nl[n]][1]
				#endif
				#if( N2G > 0 )
				- (F2[nl[n]][ind2][k] - F2[nl[n]][ind0][k]) / dx[nl[n]][2]
				#endif
				#if( N3G > 0 )
				- (F3[nl[n]][ind3][k] - F3[nl[n]][ind0][k]) / dx[nl[n]][3]
				#endif	
				+ dU_MHD1[nl[n]][ind0][k] + (1. - 2. * Y_IMEX) * dU_RAD0[nl[n]][ind0][k]);
			}

			#if STAGGERED
			U_1[B1] = 0.5*(psh[nl[n]][ind0][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1] + psh[nl[n]][index_3D(n, i + D1, j, z)][1] * gdet[nl[n]][index_2D(n, i + D1, j, z)][FACE1]);
			U_1[B2] = 0.5*(psh[nl[n]][ind0][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2] + psh[nl[n]][index_3D(n, i, j + D2, z)][2] * gdet[nl[n]][index_2D(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U_1[B3] = 0.5*(psh[nl[n]][ind0][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3] + psh[nl[n]][index_3D(n, i, j, z + D3)][3] * gdet[nl[n]][index_2D(n, i, j, z + D3)][FACE3]);
			#endif
			#endif

			PLOOP ph[nl[n]][ind0][k] = p[nl[n]][ind0][k];
			//implicit_res_solve(ph[nl[n]][ind0], U_n[nl[n]][ind0], U_1, U_1, &pflag[nl[n]][ind0], &pflag_rad[nl[n]][ind0], &geom, dU_RAD1[nl[n]][ind0], Y_IMEX*Dt, cell_size);
		}
	}
	#endif
}

void utoprim_M1_2_res(double Dt, int n){
	#if(RESISTIVE)
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U_2[NPR], dU[NPR], q;
	struct of_geom geom;
	int ind0, ind1, ind2, ind3;

	#pragma omp  parallel shared(n, gdet, p, ps, dU_MHD1, failimage, Dt, F1, F2, F3, pflag, dx, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset, nthreads, gam) private(i, j, z, k, dU, U_2, geom, ind0, ind1, ind2, ind3,q)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			get_geometry(n, i, j, z, CENT, &geom);
			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);

			q = divE_calc(ph, n, i, j, z);
			source_res(ph[nl[n]][ind0], &geom, n, i, j, z, dU, &q, Dt);

			#pragma ivdep
			PLOOP{
				U_2[k] = U_n[nl[n]][ind0][k] + Dt * (
				#if( N1G > 0 )
				- (F1[nl[n]][ind1][k] - F1[nl[n]][ind0][k]) / dx[nl[n]][1]
				#endif
				#if( N2G > 0 )
				- (F2[nl[n]][ind2][k] - F2[nl[n]][ind0][k]) / dx[nl[n]][2]
				#endif
				#if( N3G > 0 )
				- (F3[nl[n]][ind3][k] - F3[nl[n]][ind0][k]) / dx[nl[n]][3]
				#endif
				+ 0.5* (dU[k] + dU_MHD1[nl[n]][ind0][k] + dU_RAD0[nl[n]][ind0][k] +  dU_RAD1[nl[n]][ind0][k]));
			}

			#if STAGGERED
			U_2[B1] = 0.5 * (ps[nl[n]][ind0][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1] + ps[nl[n]][ind1][1] * gdet[nl[n]][index_2D(n, i + D1, j, z)][FACE1]);
			U_2[B2] = 0.5 * (ps[nl[n]][ind0][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2] + ps[nl[n]][ind2][2] * gdet[nl[n]][index_2D(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U_2[B3] = 0.5 * (ps[nl[n]][ind0][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3] + ps[nl[n]][ind3][3] * gdet[nl[n]][index_2D(n, i, j, z + D3)][FACE3]);
			#endif
			#endif
			
			//Inversion (presently without backup)
			//pflag[nl[n]][ind0] = Utoprim_3d_res(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], NEWT_TOL, BASIC, Dt);		
		}
	}
	#endif
}

void utoprim_res(double(*restrict pi[NB_LOCAL])[NPR], double(*restrict pb[NB_LOCAL])[NPR], double(*restrict pf[NB_LOCAL])[NPR], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n)
{
	#if(RESISTIVE)
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U[NPR], U0[NPR], dU[NPR], dU_RAD0[NPR], dU_RAD1[NPR], q;
	double y = 1.0 - 1.0 / sqrt(2.0);
	struct of_geom geom;
	struct of_state_res q_res;
	int ind0, ind1, ind2, ind3;

	#pragma omp  parallel shared(n,gdet, pi,pb, pf, psf, dU_s, Katm, failimage, Dt, F1, F2,F3, pflag, dx,  N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads, gam) private(i,j,z,k, geom, q_res, U, dU, ind0, ind1, ind2,ind3, q)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			get_geometry(n, i, j, z, CENT, &geom);

			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);

			q = divE_calc(pb, n, i, j, z);
			source_res(pb[nl[n]][ind0], &geom, n, i, j, z, dU,& q, Dt);
			get_state_res(pi[nl[n]][ind0], &geom, &q_res);
			primtoflux_res(pi[nl[n]][ind0], &q_res,  0, &geom, U);

			#pragma ivdep
			PLOOP{
				U[k] += Dt * (
				#if( N1G > 0 )
				- (F1[nl[n]][ind1][k] - F1[nl[n]][ind0][k]) / dx[nl[n]][1]
				#endif
				#if( N2G > 0 )
				- (F2[nl[n]][ind2][k] - F2[nl[n]][ind0][k]) / dx[nl[n]][2]
				#endif
				#if( N3G > 0 )
				- (F3[nl[n]][ind3][k] - F3[nl[n]][ind0][k]) / dx[nl[n]][3]
				#endif
				+ dU[k]);

			}
			#if STAGGERED
			U[B1] = 0.5 * (psf[nl[n]][ind0][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1] + psf[nl[n]][ind1][1] * gdet[nl[n]][index_2D(n, i + D1, j, z)][FACE1]);
			U[B2] = 0.5 * (psf[nl[n]][ind0][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2] + psf[nl[n]][ind2][2] * gdet[nl[n]][index_2D(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U[B3] = 0.5 * (psf[nl[n]][ind0][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3] + psf[nl[n]][ind3][3] * gdet[nl[n]][index_2D(n, i, j, z + D3)][FACE3]);
			#endif
			#endif

			pflag[nl[n]][ind0] = Utoprim_3d_res(U, geom.gcov, geom.gcon, geom.g, pf[nl[n]][ind0], NEWT_TOL, BASIC, Dt);
		}
	}
	#endif
}

double fluxcalc_res(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n)
{
	#if(RESISTIVE)
	#if(FRAME_TRANSFORM)
	//ndt = fluxcalc_hlld_(pr, F, dir, flag, n);
	if (rank == 0)fprintf(stderr, "Advanced Riemann solvers not implemented for resistive module \n");
	return ndt;
	#endif
	int i, j, z, k, idel, jdel, zdel, face;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR], F_HLL[NPR], U_HLL[NPR], vcon[NDIM], U_i[NPR], ptot;
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ndt, ndt_thread, dtij;
    double cmax_l_rad, cmax_r_rad, cmin_l_rad, cmin_r_rad, cmax_rad, cmin_rad;
    double ctop, ctop_rad;
	struct of_geom geom;
	struct of_state_res state_l_res, state_r_res;
	double bsq;
	int max_i, max_j, max_z;
	double val;
	ndt = 1.e9;
	int ind0, ind1, ind2;
	int fail_HLLC=0;
	int counter0 = 0;
	int counter1 = 0;
	double test;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }
	else { exit(10); }
	
		#pragma omp parallel shared(counter0,counter1,block, n_ord,n_active,n, gam, ps,t, psh,flag, pr, dq, ndt, cour, dx,dir,  F, face, idel, jdel, zdel,  N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l_res, state_r_res, F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin,cmax_l_rad, cmax_r_rad, cmin_l_rad, cmin_r_rad, cmax_rad, cmin_rad, cmax_roe, cmin_roe, ctop,ctop_rad, dtij, ind0, ind1, ind2, U_HLL, F_HLL, vcon, U_i, bsq, fail_HLLC, test, ptot)
		{
			ndt_thread = 1.e9;

			/* then evaluate slopes */
			#pragma omp for collapse(3) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)*(BS_3+2*D3)/nthreads)
			ZSLOOP3D(N1_GPU_offset[n] - D1, N1_GPU_offset[n] + BS_1 - 1 + D1, N2_GPU_offset[n] - D2, N2_GPU_offset[n] + BS_2 - 1 + D2, N3_GPU_offset[n] - D3, N3_GPU_offset[n] + BS_3 - 1 + D3){
				// #pragma ivdep
				ind0 = index_3D(n, i, j, z);
				ind1 = index_3D(n, i - idel, j - jdel, z - zdel);
				ind2 = index_3D(n, i + idel, j + jdel, z + zdel);
				PLOOP{
					dq[nl[n]][ind0][k] = slope_lim(pr[nl[n]][ind1][k], pr[nl[n]][ind0][k], pr[nl[n]][ind2][k]);
				}
			}

			#pragma omp for collapse(3) schedule(static,(BS_1+jdel+zdel+1)*(BS_2+idel+zdel+1)*(BS_3+idel+jdel+1)/nthreads)
			ZSLOOP((N1_GPU_offset[n] - jdel - zdel)*D1, (N1_GPU_offset[n] + BS_1)*D1, (N2_GPU_offset[n] - idel - zdel)*D2, (N2_GPU_offset[n] + BS_2)*D2) 	{
				for (z = (N3_GPU_offset[n] - idel - jdel)*D3; z <= (N3_GPU_offset[n] + BS_3)*D3; z++){
					get_geometry(n, i, j, z, face, &geom);
					/* this avoids problems on the pole */
					ind0 = index_3D(n, i, j, z);
					ind1 = index_3D(n, i - idel, j - jdel, z - zdel);

					#pragma ivdep
					PLOOP{
						p_l[k] = pr[nl[n]][ind1][k] + 0.5*dq[nl[n]][ind1][k];
						p_r[k] = pr[nl[n]][ind0][k] - 0.5*dq[nl[n]][ind0][k];
						#if(STAGGERED)
						if ((dir == 1 && k == B1)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 2 && k == B2)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 3 && k == B3)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						#endif
					}

					#if(STAGGERED)
					if ((dir == 2) && ((j == 0 && (block[n][AMR_NBR1]<0 || block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3)) || (j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2])) && (block[n][AMR_NBR3]<0 || block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3)))){
						p_r[B1] = 0.;
						p_l[B1] = 0.;
					}
					#endif

					get_state_res(p_l, &geom, &state_l_res);
					get_state_res(p_r, &geom, &state_r_res);
				
					primtoflux_res(p_l, &state_l_res, dir, &geom, F_l);
					primtoflux_res(p_r, &state_r_res, dir, &geom, F_r);

					primtoflux_res(p_l, &state_l_res, 0, &geom, U_l);
					primtoflux_res(p_r, &state_r_res, 0, &geom, U_r);

					vchar_res(&geom, dir, &cmax_l, &cmin_l);
					vchar_res(&geom, dir, &cmax_r, &cmin_r);

					cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
					cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
					ctop = MY_MAX(cmax, cmin);

					#if(DO_IMEX)
					#pragma ivdep
					if (flag == 1) {
						for (k = 0; k < NPR; k++) {
							#if(HLLF)
							F[nl[n]][ind0][k] = 0.5 * (F[nl[n]][ind0][k] + (cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL));
							#else
							F[nl[n]][ind0][k] = 0.5 * (F[nl[n]][ind0][k] + 0.5 * (F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k])));
							#endif
						}
					}
					else {
						for (k = 0; k < NPR; k++) {
							#if(HLLF)
							F[nl[n]][ind0][k] = (cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL);
							#else
							F[nl[n]][ind0][k] = 0.5 * (F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k]));
							#endif
						}
					}
					#else
					#pragma ivdep
					for (k = 0; k <NPR; k++) {
						#if(HLLF)
						F[nl[n]][ind0][k] = (cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL);
						#else
						F[nl[n]][ind0][k] = 0.5 * (F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k]));
						#endif
					}
					#endif

					/* evaluate restriction on timestep */
					cmax = MY_MAX(cmax, cmin);

					dtij = cour*dx[nl[n]][dir] / cmax;
					if (dtij < ndt_thread) {
						ndt_thread = dtij;
						#if(!TRANS_BOUND && !CARTESIAN)
						if (dir == 2 && (j == 0 || j == N2 * pow(1+REF_2,block[n][AMR_LEVEL]))) {
                            PLOOP F[nl[n]][ind0][k] = 0.;
						}
						#endif
					}
				}
			}

			#pragma omp critical
			{
				if (ndt_thread < ndt){
					ndt = ndt_thread;
				}
			}
		}
	return(ndt);
	#endif
}
