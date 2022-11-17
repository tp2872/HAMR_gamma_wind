#include "include.h"
#include "decs_MPI.h"
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
	int i, j, k, n, u;

	if (rank == 0) fprintf(stderr, "h");

	for (u = 0; u < 2*AMR_MAXTIMELEVEL; u++){
		set_prestep();
		ndt = advance(0);

		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  fixup(p, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup(ph, n_ord[n]);
		}

		/*for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup_utoprim(p, n_ord[n]);  //Fix the failure points using interpolation and updated ghost zone values
			else fixup_utoprim(ph, n_ord[n]);
			pflag[n_ord[n]][index_3D(n_ord[n] ,N1_GPU_offset[n_ord[n]] - N1G, N2_GPU_offset[n_ord[n]] - N2G, N3_GPU_offset[n_ord[n]] - N3G)] = 0;
		}*/

		#if(RAD_M1)
		for (n = 0; n < n_active; n++){
			//if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup_utoprim_rad(p, n_ord[n]);  //Fix the failure points using interpolation and updated ghost zone values
			//else fixup_utoprim_rad(ph, n_ord[n]);
			//pflag_rad[n_ord[n]][index_3D(n_ord[n] ,N1_GPU_offset[n_ord[n]] - N1G, N2_GPU_offset[n_ord[n]] - N2G, N3_GPU_offset[n_ord[n]] - N3G)] = 0;
		}
		#endif

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

	//Set mass density scale if necessary
	#if(CALC_MDOT)
	set_mass_density_scale(&mass_density_scale_cpu);
	#endif

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
double advance(int flag)
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

	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_0(dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
			fixup(p, n_ord[n]);
		}
	}
	#endif

	#if(N1G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][1] = fluxcalc(ph, F1, 1, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][1] = fluxcalc(p, F1, 1, 0, n_ord[n]);
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
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[nl[n_ord[n]]][2] = fluxcalc(ph, F2, 2, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][2] = fluxcalc(p, F2, 2, 0, n_ord[n]);
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
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[nl[n_ord[n]]][3] = fluxcalc(ph, F3, 3, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[nl[n_ord[n]]][3] = fluxcalc(p, F3, 3, 0, n_ord[n]);
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

	#if(!TRANS_BOUND && (SPHERICAL || SPHERICAL_GR))
	for (n = 0; n < n_active; n++) if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fix_flux(F1, F2, F3, n_ord[n]);
	#endif

	#if(!STAGGERED)
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_ct(F1, F2, F3, n_ord[n]);
	#else
	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1_M1_2(ph, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1(p, n_ord[n]);
	}
	#else
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1(ph, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1(p, n_ord[n]);
	}
	#endif

	const_transport_bound();

	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2(ps, ps, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2(ps, psh, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#else
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  const_transport2(ps, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2(ps, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif
	#endif

	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_2(dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			utoprim_M1_1(dt* (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		}
	}
	#else
	for (n = 0; n < n_active; n++){
        if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  utoprim(p, ph, p, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) utoprim(p, p, ph, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
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

#if(RAD_M1)
void utoprim_M1_0(double Dt, int n)
{
	int i, j, z, k, ind0;
	double cell_size;
	struct of_geom geom;
	struct of_state q;
	struct of_state_rad q_rad;
	double gamma_g;

	#pragma omp  parallel shared(n, p, Dt, pflag, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset) private(i, j, z, k, geom,  q, q_rad, ind0, cell_size, gamma_g)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n]-N1G, N1_GPU_offset[n] + BS_1 + N1G - 1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G - 1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 +N3G - 1) {
			k = 1;
			//if (((i < N1_GPU_offset[n] - D1) && ((j < N2_GPU_offset[n] - D2))) || ((i < N1_GPU_offset[n] - D1) && ((j > N2_GPU_offset[n] + BS_2))) || (((i > N1_GPU_offset[n] + BS_1) && ((j < N2_GPU_offset[n] - D2)))) || ((i > N1_GPU_offset[n] + BS_1) && ((j > N2_GPU_offset[n] + BS_2)))) k = 0;
			//if (((i < N1_GPU_offset[n] - D1) && ((z < N3_GPU_offset[n] - D3))) || ((i < N1_GPU_offset[n] - D1) && ((z > N3_GPU_offset[n] + BS_3))) || (((i > N1_GPU_offset[n] + BS_1) && ((z < N3_GPU_offset[n] - D3)))) || ((i > N1_GPU_offset[n] + BS_1) && ((z > N3_GPU_offset[n] + BS_3)))) k = 0;
			//if (((j < N2_GPU_offset[n] - D2) && ((z < N3_GPU_offset[n] - D3))) || ((j < N2_GPU_offset[n] - D2) && ((z > N3_GPU_offset[n] + BS_3))) || (((j > N2_GPU_offset[n] + BS_2) && ((z < N3_GPU_offset[n] - D3)))) || ((j > N2_GPU_offset[n] + BS_2) && ((z > N3_GPU_offset[n] + BS_3)))) k = 0;

			if (k == 1) {
				ind0 = index_3D(n, i, j, z);
				get_geometry(n, i, j, z, CENT, &geom);
				get_state(p[nl[n]][ind0], &geom, &q);
				get_state_rad(p[nl[n]][ind0], &geom, &q_rad);		
				#if(TWO_T)
				double fel = calc_delta(p[nl[n]][ind0], dot(q.bcon, q.bcov));
				gamma_g = calc_gamma_gas_prim(p[nl[n]][ind0]);
				#endif				
				primtoflux(p[nl[n]][ind0], &q, &q_rad, 0, &geom, U_n[nl[n]][ind0]
					#if(TWO_T)
					, gamma_g
					#endif
				);

				cell_size = MY_MAX(MY_MAX(dx[nl[n]][1] * sqrt(geom.gcov[1][1]), dx[nl[n]][2] * sqrt(geom.gcov[2][2])), dx[nl[n]][3] * sqrt(geom.gcov[3][3]));
				implicit_rad_solve(p[nl[n]][ind0], U_n[nl[n]][ind0], U_n[nl[n]][ind0], U_0[nl[n]][ind0], &pflag[nl[n]][ind0], &pflag_rad[nl[n]][ind0], &geom, dU_RAD0[nl[n]][ind0], Dt * Y_IMEX, cell_size
				#if(TWO_T)
					, fel
				#endif
				);
			}
		}
	}
}

void utoprim_M1_1(double Dt, int n){
	int i, j, z, k;
	double cell_size, dU_MHD[NPR];
	struct of_geom geom;
	int ind0, ind1, ind2, ind3;
	double gamma_g;

	#pragma omp  parallel shared(n, gdet, psh, dU_MHD1, Dt, F1, F2, F3, dx, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset, nthreads, gam) private(i, j, z, k, geom, dU_MHD, ind0, ind1, ind2, ind3, cell_size, gamma_g)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			get_geometry(n, i, j, z, CENT, &geom);

			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);

			#if(TWO_T)
			struct of_state q;
			get_state(p[nl[n]][ind0], &geom, &q);
			double fel = calc_delta(p[nl[n]][ind0], dot(q.bcon, q.bcov));
			gamma_g = calc_gamma_gas_prim(p[nl[n]][ind0]);
			#endif	
			source(p[nl[n]][ind0], &geom, n, i, j, z, dU_MHD, Dt
				#if(TWO_T)
				, gamma_g
				#endif
			);

			PLOOP{
				U_1[nl[n]][ind0][k] = (((3.0 * Y_IMEX - 1.0) / Y_IMEX) * U_n[nl[n]][ind0][k] + ((1.0 - 2.0 * Y_IMEX) / Y_IMEX) * U_0[nl[n]][ind0][k])  + Dt * (
				#if( N1G > 0 )
				- (F1[nl[n]][ind1][k] - F1[nl[n]][ind0][k]) / dx[nl[n]][1]
				#endif
				#if( N2G > 0 )
				- (F2[nl[n]][ind2][k] - F2[nl[n]][ind0][k]) / dx[nl[n]][2]
				#endif
				#if( N3G > 0 )
				- (F3[nl[n]][ind3][k] - F3[nl[n]][ind0][k]) / dx[nl[n]][3]
				#endif	
				+ dU_MHD[k]);
			}

			#if STAGGERED
			U_1[nl[n]][ind0][B1] = 0.5*(psh[nl[n]][ind0][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1] + psh[nl[n]][index_3D(n, i + D1, j, z)][1] * gdet[nl[n]][index_2D(n, i + D1, j, z)][FACE1]);
			U_1[nl[n]][ind0][B2] = 0.5*(psh[nl[n]][ind0][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2] + psh[nl[n]][index_3D(n, i, j + D2, z)][2] * gdet[nl[n]][index_2D(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U_1[nl[n]][ind0][B3] = 0.5*(psh[nl[n]][ind0][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3] + psh[nl[n]][index_3D(n, i, j, z + D3)][3] * gdet[nl[n]][index_2D(n, i, j, z + D3)][FACE3]);
			#endif
			#endif

			PLOOP ph[nl[n]][ind0][k] = p[nl[n]][ind0][k];
			cell_size = MY_MAX(MY_MAX(dx[nl[n]][1] * sqrt(geom.gcov[1][1]), dx[nl[n]][2] * sqrt(geom.gcov[2][2])), dx[nl[n]][3] * sqrt(geom.gcov[3][3]));
			implicit_rad_solve(ph[nl[n]][ind0], U_n[nl[n]][ind0], U_1[nl[n]][ind0], U_1[nl[n]][ind0], &pflag[nl[n]][ind0], &pflag_rad[nl[n]][ind0], &geom, dU_RAD1[nl[n]][ind0], Y_IMEX*Dt, cell_size
				#if(TWO_T)
				, fel
				#endif	
			);
		}
	}
}

void utoprim_M1_2(double Dt, int n){
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U_2[NPR], dU[NPR], gamma_g;
	struct of_geom geom;
	int ind0, ind1, ind2, ind3;
	double fel;

	#pragma omp  parallel shared(n, gdet, p, ps, dU_MHD1, failimage, Dt, F1, F2, F3, pflag, dx, N1_GPU_offset, N2_GPU_offset, N3_GPU_offset, nthreads, gam) private(i, j, z, k, fel, dU, U_2, geom, ind0, ind1, ind2, ind3, gamma_g)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			get_geometry(n, i, j, z, CENT, &geom);
			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);
			#if(TWO_T)
			fprintf(stderr, "calc_delta not implemented! \n");
			gamma_g = calc_gamma_gas_prim(ph[nl[n]][ind0]);
			#else
			gamma_g = GAMMA;
			#endif	
			source(ph[nl[n]][ind0], &geom, n, i, j, z, dU, Dt
				#if(TWO_T)
				, gamma_g
				#endif
			);

			#pragma ivdep
			PLOOP{
				U_2[k] = 0.5 * (U_n[nl[n]][ind0][k] + U_1[nl[n]][ind0][k]) + Dt * (
				#if( N1G > 0 )
				- 0.5 * (F1[nl[n]][ind1][k] - F1[nl[n]][ind0][k]) / dx[nl[n]][1]
				#endif
				#if( N2G > 0 )
				- 0.5 * (F2[nl[n]][ind2][k] - F2[nl[n]][ind0][k]) / dx[nl[n]][2]
				#endif
				#if( N3G > 0 )
				- 0.5 * (F3[nl[n]][ind3][k] - F3[nl[n]][ind0][k]) / dx[nl[n]][3]
				#endif
				+ (0.5 * dU[k] + Y_IMEX * dU_RAD0[nl[n]][ind0][k] + 0.5 * (1.0 - Y_IMEX) * dU_RAD1[nl[n]][ind0][k]));
			}

			#if STAGGERED
			U_2[B1] = 0.5 * (ps[nl[n]][ind0][1] * gdet[nl[n]][index_2D(n, i, j, z)][FACE1] + ps[nl[n]][ind1][1] * gdet[nl[n]][index_2D(n, i + D1, j, z)][FACE1]);
			U_2[B2] = 0.5 * (ps[nl[n]][ind0][2] * gdet[nl[n]][index_2D(n, i, j, z)][FACE2] + ps[nl[n]][ind2][2] * gdet[nl[n]][index_2D(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U_2[B3] = 0.5 * (ps[nl[n]][ind0][3] * gdet[nl[n]][index_2D(n, i, j, z)][FACE3] + ps[nl[n]][ind3][3] * gdet[nl[n]][index_2D(n, i, j, z + D3)][FACE3]);
			#endif
			#endif

			#if(NEWMAN)
			pflag[nl[n]][ind0] = Utoprim_NM(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], NEWT_TOL, BASIC
				#if(TWO_T)
				, fel
				#endif
			);
			#else
			pflag[nl[n]][ind0] = Utoprim_2d(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], NEWT_TOL, BASIC
				#if(TWO_T)
				, fel
				#endif
			);
			#endif
			#if( DO_FONT_FIX ) 
			if (pflag[nl[n]][ind0]) {
				failimage[nl[n]][ind0][0]++;
				pflag[nl[n]][ind0] = Utoprim_1dvsq2fix1(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], NEWT_TOL, BASIC
					#if(TWO_T)
					, fel
					#endif
				);
				if (pflag[nl[n]][ind0]) {
					failimage[nl[n]][ind0][1]++;
					pflag[nl[n]][ind0] = Utoprim_1dfix1(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], NEWT_TOL, BASIC
						#if(TWO_T)
						, fel
						#endif
					);
					if (pflag[nl[n]][ind0]) {
						failimage[nl[n]][ind0][2]++;
						pflag[nl[n]][index_3D(n, N1_GPU_offset[n] - N1G, N2_GPU_offset[n] - N2G, N3_GPU_offset[n] - N3G)] = 100;
					}
				}
			}
			#endif
			pflag_rad[nl[n]][ind0] = Rtoprim(U_2, geom.gcov, geom.gcon, geom.g, p[nl[n]][ind0], BASIC);
		}
	}
}
#endif

void utoprim(double(*restrict pi[NB_LOCAL])[NPR], double(*restrict pb[NB_LOCAL])[NPR], double(*restrict pf[NB_LOCAL])[NPR], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n)
{
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U[NPR], U0[NPR], dU[NPR], dU_RAD0[NPR], dU_RAD1[NPR], gamma_g;
	double y = 1.0 - 1.0 / sqrt(2.0);
	struct of_geom geom;
	struct of_state q;
	struct of_state_rad q_rad;
	int ind0, ind1, ind2, ind3;
	double fel;

	#pragma omp  parallel shared(n,gdet, pi,pb, pf, psf, dU_s, Katm, failimage, Dt, F1, F2,F3, pflag, dx,  N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads, gam) private(i,j,z,k, fel, geom, q,q_rad, U, dU, ind0, ind1, ind2,ind3, gamma_g)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			get_geometry(n, i, j, z, CENT, &geom);

			ind0 = index_3D(n, i, j, z);
			ind1 = index_3D(n, i + D1, j, z);
			ind2 = index_3D(n, i, j + D2, z);
			ind3 = index_3D(n, i, j, z + D3);

			#if(TWO_T)
			gamma_g = calc_gamma_gas_prim(pb[nl[n]][ind0]);
			#else
			gamma_g = GAMMA;
			#endif
			source(pb[nl[n]][ind0], &geom, n, i, j, z, dU, Dt
				#if(TWO_T)
				, gamma_g
				#endif
			);

			get_state(pi[nl[n]][ind0], &geom, &q);
			#if(RAD_M1)
			get_state_rad(pi[nl[n]][ind0], &geom, &q_rad);
			#endif
			#if(TWO_T)
			gamma_g = calc_gamma_gas_prim(pi[nl[n]][ind0]);
			fel = calc_delta(pb[nl[n]][ind0], dot(q.bcon, q.bcov));
			#else
			gamma_g = GAMMA;
			#endif
			primtoflux(pi[nl[n]][ind0], &q, &q_rad, 0, &geom, U
				#if(TWO_T)
				, gamma_g
				#endif
			);

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

			#if(RAD_M1)
			double U_0[NPR];
			int pflag_local, pflag_rad_local;
			PLOOP dU[k] = 0.;

			//Perform implicit solve
			double cell_size = MY_MAX(MY_MAX(dx[nl[n]][1] * sqrt(geom.gcov[1][1]), dx[nl[n]][2] * sqrt(geom.gcov[2][2])), dx[nl[n]][3] * sqrt(geom.gcov[3][3]));
			implicit_rad_solve(pf[nl[n]][ind0], U, U, U_0, &pflag[nl[n]][ind0], &pflag_rad[nl[n]][ind0], &geom, dU, Dt, cell_size
				#if(TWO_T)
				, fel
				#endif
			);
			#else


			#if(NEWMAN)
			pflag[nl[n]][ind0] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf[nl[n]][ind0], NEWT_TOL, BASIC
				#if(TWO_T)
				, fel
				#endif
			);
			#else
			pflag[nl[n]][ind0] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf[nl[n]][ind0], NEWT_TOL, BASIC
				#if(TWO_T)
				, fel
				#endif
			);
			#endif

			/*#if(DO_FONT_FIX)
			if (pflag[nl[n]][ind0]) {
				failimage[nl[n]][ind0][0]++;
				#if DOKTOT
				pflag[nl[n]][ind0] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf[nl[n]][ind0], NEWT_TOL, BASIC
				#if(TWO_T)
				, fel
				#endif
				);
				#endif
				if (pflag[nl[n]][ind0]) {
					failimage[nl[n]][ind0][1]++;
					pflag[nl[n]][ind0] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf[nl[n]][ind0], NEWT_TOL, BASIC
					#if(TWO_T)
					, fel
					#endif
					);
					if (pflag[nl[n]][ind0]) {
						pflag[nl[n]][index_3D(n, N1_GPU_offset[n] - N1G, N2_GPU_offset[n] - N2G, N3_GPU_offset[n] - N3G)] = 100;
						failimage[nl[n]][ind0][2]++;
					}
				}
			}
			#endif*/
			#endif
		}
	}
}


/***********************************************************************************************/
/***********************************************************************************************
fluxcalc():
---------
-- sets the numerical fluxes, avaluated at the cell boundaries using the slope limiter
slope_lim();

-- only has HLL and Lax-Friedrichs  approximate Riemann solvers implemented;

***********************************************************************************************/
double fluxcalc(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n)
{
	#if(FRAME_TRANSFORM)
	ndt = fluxcalc_hlld(pr, F, dir, flag, n);
	return ndt;
	#endif
	int i, j, z, k, idel, jdel, zdel, face;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR], F_HLL[NPR], U_HLL[NPR], vcon[NDIM], U_i[NPR], ptot;
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ndt, ndt_thread, dtij;
    double cmax_l_rad, cmax_r_rad, cmin_l_rad, cmin_r_rad, cmax_rad, cmin_rad, gamma_g;
    double ctop, ctop_rad;
	struct of_geom geom;
	struct of_state state_l, state_r, state_roe, qi;
    struct of_state_rad state_l_rad, state_r_rad;
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
		#pragma omp parallel shared(counter0,counter1,block, n_ord,n_active,n, gam, ps,t, psh,flag, pr, dq, ndt, cour, dx,dir,  F, face, idel, jdel, zdel,  N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l, state_r, state_l_rad, state_r_rad,state_roe, gamma_g, F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin,cmax_l_rad, cmax_r_rad, cmin_l_rad, cmin_r_rad, cmax_rad, cmin_rad, cmax_roe, cmin_roe, ctop,ctop_rad, dtij, ind0, ind1, ind2, U_HLL, F_HLL, qi, vcon, U_i, bsq, fail_HLLC, test, ptot)
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

					get_state(p_l, &geom, &state_l);
					get_state(p_r, &geom, &state_r);
					#if(RAD_M1)
					get_state_rad(p_l, &geom, &state_l_rad);
					get_state_rad(p_r, &geom, &state_r_rad);
					#endif

					#if(TWO_T)
					gamma_g = calc_gamma_gas_prim(p_l);
					if (gamma_g < 0.0) fprintf(stderr, "1: (%d, %d, %d): %f %f %f %f \n", i, j, z, log10(pr[nl[n]][ind1][ENTRE]), log10(pr[nl[n]][ind0][k]), log10(pr[nl[n]][index_3D(n, i + idel, j + jdel, z + zdel)][k]), log10(p_l[ENTRI]));
					#else
					gamma_g = GAMMA;
					#endif
					primtoflux(p_l, &state_l, &state_l_rad, dir, &geom, F_l
						#if(TWO_T)
						, gamma_g
						#endif
					);
					primtoflux(p_l, &state_l, &state_l_rad, 0, &geom, U_l
						#if(TWO_T)
						, gamma_g
						#endif
					);
					vchar(p_l, &state_l, &geom, dir, &cmax_l, &cmin_l
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#if(RAD_M1)
					vchar_rad(p_l, &state_l, &state_l_rad, &geom, dir, &cmax_l_rad, &cmin_l_rad, dx[nl[n]][dir]
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					#if(TWO_T)
					gamma_g = calc_gamma_gas_prim(p_r);
					if (gamma_g < 0.0) fprintf(stderr, "2: (%d, %d, %d): %f %f %f %f \n", i, j, z, log10(pr[nl[n]][ind1][ENTRE]), log10(pr[nl[n]][ind0][k]), log10(pr[nl[n]][index_3D(n, i + idel, j + jdel, z + zdel)][k]), log10(p_l[ENTRI]));
					#else
					gamma_g = GAMMA;
					#endif
					primtoflux(p_r, &state_r, &state_r_rad, dir, &geom, F_r
						#if(TWO_T)
						, gamma_g
						#endif
					);
					primtoflux(p_r, &state_r, &state_r_rad, 0, &geom, U_r
						#if(TWO_T)
						, gamma_g
						#endif
					);
					vchar(p_r, &state_r, &geom, dir, &cmax_r, &cmin_r
						#if(TWO_T)
						, gamma_g
						#endif
					);

					#if(RAD_M1)
					vchar_rad(p_r, &state_r, &state_r_rad, &geom, dir, &cmax_r_rad, &cmin_r_rad, dx[nl[n]][dir]
						#if(TWO_T)
						, gamma_g
						#endif
					);

					//Find radiation wavespeeds
					cmax_rad = fabs(MY_MAX(MY_MAX(0., cmax_l_rad), cmax_r_rad));
					cmin_rad = fabs(MY_MAX(MY_MAX(0., -cmin_l_rad), -cmin_r_rad));
					ctop_rad = MY_MAX(cmax_rad, cmin_rad);
					#endif

					//Find MHD wavespeeds
					cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
					cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
					ctop = MY_MAX(cmax, cmin);

					#if(RAD_M1)
					for (k = 0; k < NPR; k++) {
						if (k == UU_RAD || k == U1_RAD || k == U2_RAD || k == U3_RAD || k == PHOTON) {
							F[nl[n]][ind0][k] = 0.5 * (F_l[k] + F_r[k] - ctop_rad * (U_r[k] - U_l[k]));
						}
						else {
							#if(HLLF)
							F[nl[n]][ind0][k] = (cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL);
							#else
							F[nl[n]][ind0][k] = 0.5 * (F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k]));
							#endif
						}
					}
					#else
					for (k = 0; k < NPR; k++) {
						#if(HLLF)
						F[nl[n]][ind0][k] = (cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL);
						#else
						F[nl[n]][ind0][k] = 0.5 * (F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k]));
						#endif
					}
					#endif

					/* evaluate restriction on timestep */
					cmax = MY_MAX(cmax, cmin);
                    #if(RAD_M1)
                    cmax_rad = MY_MAX(cmax_rad, cmin_rad);
                    cmax = MY_MAX(cmax, cmax_rad);
                    #endif
					dtij = cour*dx[nl[n]][dir] / cmax;
					if (dtij < ndt_thread) {
						ndt_thread = dtij;
						#if(!TRANS_BOUND && (SPHERICAL || SPHERICAL_GR))
						if (dir == 2 && (j == 0 || j == N2 * pow(1+REF_2,block[n][AMR_LEVEL2]))) {
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
}

void GPU_step_ch()
{
	double ndt, inmsg;
	int i, j, z, k, n, uu;

	if (rank == 0)fprintf(stderr, "h");

	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
	}
	for (uu = 0; uu < 2 * AMR_MAXTIMELEVEL; uu++){
		set_prestep();
		ndt = advance_GPU();   /* time step primitive variables to the half step */

		//Post-stepping when having 2nd order time accuracy at boundary
		#if(PRESTEP2)
		prestep_bound();
		#endif

		//Set boundary conditions at end of timestep after correction step to fluxes and electric fields
		GPU_boundprim(0);    

		nstep++;
		#if(PRESTEP || PRESTEP2)
		for (n = 0; n < n_active; n++){
			if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && (block[n_ord[n]][AMR_PRESTEP] != 0))block[n_ord[n]][AMR_PRESTEP] = 0;
			else if (block[n_ord[n]][AMR_PRESTEP] == 1)block[n_ord[n]][AMR_PRESTEP] = 2;
		}
		#endif
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	if (rank == 0){
		fprintf(stderr, "f");
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		if(rank==0) fprintf(stderr, "timestep too small\n");
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

	//Set mass density scale if necessary
	#if(CALC_MDOT)
	set_mass_density_scale(&mass_density_scale_cpu);
	#endif
}

double advance_GPU(void)
{
	int i, n, flag;
	double timestep, temp;
	gpu = 1;
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 0){
		ndt1 = ndt2 = ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[nl[n_ord[n]]][0] = bdt[nl[n_ord[n]]][1] = bdt[nl[n_ord[n]]][2] = bdt[nl[n_ord[n]]][3] = 1e9;
		}
	}

	for (n = 0; n < n_active; n++){
		prestep_half[nl[n_ord[n]]] = (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) < block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
		prestep_full[nl[n_ord[n]]] = (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) < 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) >  block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
	}

	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (prestep_full[nl[n_ord[n]]] == 1) {
		}
		else if (prestep_half[nl[n_ord[n]]] == 1) {
			GPU_Utoprim_M1_0(n_ord[n], dt * (double)block[n_ord[n]][AMR_TIMELEVEL]);
		}
	}
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(1, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(1, n_ord[n]);
	}
	#endif
	#endif

	#if(N3G>0)		
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1){
			GPU_fluxcalc2D(3, 1, n_ord[n]);
			#if(RECONSTRUCT_INTERNAL)
				#if(N_LEVELS_1D_INT>0)
				GPU_reconstruct_internal(1, n_ord[n]);
				#endif
			#endif
		}
		else if (prestep_half[nl[n_ord[n]]] == 1){
			GPU_fluxcalc2D(3, 0, n_ord[n]);
			#if(RECONSTRUCT_INTERNAL)
				#if(N_LEVELS_1D_INT>0)
				GPU_reconstruct_internal(0, n_ord[n]);
				#endif
			#endif
		}
	}

	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[nl[n_ord[n]]][3] = fluxcalc_GPU(n_ord[n], 3);
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
				ndt3 = MY_MIN(ndt3, bdt[nl[n_ord[n]]][3]);
			}
			else{
				ndt3 = MY_MIN(ndt3, bdt[nl[n_ord[n]]][3] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt3 = 1e9;
	#endif
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		#if(N3G>0)
		flux_send3(F3, BufferF3_1, n_ord[n]);
		#endif
	}

	#if(N2G>0)
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_fluxcalc2D(2, 1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_fluxcalc2D(2, 0, n_ord[n]);
	}

	//read_time_GPU();
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt2 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[nl[n_ord[n]]][2] = fluxcalc_GPU(n_ord[n], 2);
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
				ndt2 = MY_MIN(ndt2, bdt[nl[n_ord[n]]][2]);
			}
			else{
				ndt2 = MY_MIN(ndt2, bdt[nl[n_ord[n]]][2] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt2 = 1e9;
	#endif
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		flux_send2(F2, BufferF2_1, n_ord[n]);
	}

	#if(N1G>0)
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_fluxcalc2D(1, 1, n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_fluxcalc2D(1, 0, n_ord[n]);
	}

	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt1 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[nl[n_ord[n]]][1] = fluxcalc_GPU(n_ord[n], 1);
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1) {
				ndt1 = MY_MIN(ndt1, bdt[nl[n_ord[n]]][1]);
			}
			else{
				ndt1 = MY_MIN(ndt1, bdt[nl[n_ord[n]]][1] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt1 = 1e9;
	#endif
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		flux_send1(F1, BufferF1_1, n_ord[n]);
	}

	gpu = 1;
	rc = 0;

	#if(PRESTEP)
	set_iprobe(0, &flag);
	do{
		//For last timestep synchronize electric fields immediately
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif	
		for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif	
			flux_rec1(F1, BufferF1_1, n_ord[n], 5);
			flux_rec2(F2, BufferF2_1, n_ord[n], 5);
			#if(N3G>0)
			flux_rec3(F3, BufferF3_1, n_ord[n], 5);
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);

	//For first timestep do not synchronize electrice fields 
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif	
		flux_rec1(F1, BufferF1_1, n_ord[n], 2);
		flux_rec2(F2, BufferF2_1, n_ord[n], 2);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 2);
		#endif
	}
	#elif(!PRESTEP2)
	set_iprobe(0, &flag);
	do {
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif	
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			#if(N3G>0)
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);
			#endif
	}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	#endif 

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomF \n");
	#if(!TRANS_BOUND && (SPHERICAL || SPHERICAL_GR))
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) if (prestep_full[nl[n_ord[n]]] == 1 || prestep_half[nl[n_ord[n]]] == 1) GPU_fix_flux(n_ord[n]);
	#endif
	#if(STAGGERED)
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_consttransport1(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_consttransport1(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}

	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_consttransport2_M1_2(1, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_consttransport2(0, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#else
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++) {
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_consttransport2(1, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_consttransport2(0, 0.5 * dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif

	rc = 0;
	GPU_consttransport_bound();
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE \n");

	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_consttransport3(1, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_consttransport3(0, dt * (double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#else
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1) GPU_consttransport3(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[nl[n_ord[n]]] == 1) GPU_consttransport3(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif
	#else
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct1(n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct2(n_ord[n]);
	#endif

	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static, n_active/nthreads) private(n,status,timestep)
	#endif
	#if(RAD_M1 && DO_IMEX)
	for (n = 0; n < n_active; n++) {
		if (prestep_full[nl[n_ord[n]]] == 1) {
			GPU_Utoprim_M1_2(n_ord[n], dt * (double)block[n_ord[n]][AMR_TIMELEVEL]);
		}
		else if (prestep_half[nl[n_ord[n]]] == 1) {
			GPU_Utoprim_M1_1(n_ord[n], dt * (double)block[n_ord[n]][AMR_TIMELEVEL]);
		}
	}
	#else
	for (n = 0; n < n_active; n++){
		if (prestep_full[nl[n_ord[n]]] == 1){
			GPU_fixup(1, n_ord[n], dt * (double)block[n_ord[n]][AMR_TIMELEVEL]);
			//GPU_fixuputoprim(1, n_ord[n]);
			//GPU_fixuputoprim_rad(1, n_ord[n]);
		}
		else if (prestep_half[nl[n_ord[n]]] == 1 ){
			GPU_fixup(0, n_ord[n], 0.5*dt* (double)block[n_ord[n]][AMR_TIMELEVEL]);
			//GPU_fixuputoprim(0, n_ord[n]);
			//GPU_fixuputoprim_rad(0, n_ord[n]);
		}
	}
	#endif

	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
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
	}

	//ndt = defcon * 1. / (1. / ndt1 + 1. / ndt2 + 1. / ndt3);
	return defcon * ndt;
}

void benchmark_GPU(int n)
{
	#if(RAD_M1 || RESISTIVE)
	int i;
	gpu = 1;
	double start, end;
	
	//Synchronize GPU
	gpuDeviceSynchronize();
	start = get_wall_time();

	for (i = 0; i < 100; i++) {
		#if(RAD_M1 && DO_IMEX)
		if(i%2==-10) GPU_Utoprim_M1_0(n, dt * (double)block[n][AMR_TIMELEVEL]); //do not use
		#endif

		#if(N3G>0)		
		GPU_fluxcalc2D(3, 0, n);
		#endif

		#if(N2G>0)
		GPU_fluxcalc2D(2, 0, n);
		#endif

		#if(N1G>0)
		GPU_fluxcalc2D(1, 0, n);
		#endif

		#if(!TRANS_BOUND && (SPHERICAL || SPHERICAL_GR))
		GPU_fix_flux(n);
		#endif

		#if(STAGGERED)
			GPU_consttransport1(0, 0.5 * dt * (double)block[n][AMR_TIMELEVEL], n);
			#if(RAD_M1 && DO_IMEX)
			GPU_consttransport2(0, dt * (double)block[n][AMR_TIMELEVEL], n);
			#else
			GPU_consttransport2(0, 0.5 * dt * (double)block[n][AMR_TIMELEVEL], n);
			#endif

			#if(RAD_M1 && DO_IMEX)
			GPU_consttransport3(0, dt * (double)block[n][AMR_TIMELEVEL], n);
			#else
			GPU_consttransport3(0, 0.5 * dt * (double)block[n][AMR_TIMELEVEL], n);
			#endif
		#else
		GPU_flux_ct1(n);
		GPU_flux_ct2(n);
		#endif

		#if(RAD_M1 && DO_IMEX)
		GPU_Utoprim_M1_1(n, dt * (double)block[n][AMR_TIMELEVEL]);
		#else
		GPU_fixup(0, n, 0.5 * dt * (double)block[n][AMR_TIMELEVEL]);
		#endif
	}
	gpuDeviceSynchronize();
	end = get_wall_time();

	bench_time[n] = (double)(end - start);
	#else
	bench_time[n] = 1.0;
	#endif
}

/*Used for debugging. Compares output from CPU version to output from GPU version*/
void step_ch_debug()
{
	#if (GPU_ENABLED==1)
	double ndt=0., inmsg;
	int i, j, z, k, n, u;
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif
	//ndt = advance_GPU(0.5*dt, 0);   /* time step primitive variables to the half step */
	//GPU_fixup(0);
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(0);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */

	// Danat: GPU evolve
	for (n = 0; n < n_active; n++) {
		block[n_ord[n]][AMR_PRESTEP] = 0;
	}

	for (u = 0; u < 2 * AMR_MAXTIMELEVEL; u++) {
		set_prestep();
		ndt = advance_GPU();   /* time step primitive variables to the half step */

		//Post-stepping when having 2nd order time accuracy at boundary
		#if(PRESTEP2)
		prestep_bound();
		#endif

		//Set boundary conditions at end of timestep after correction step to fluxes and electric fields
		GPU_boundprim(0);

		nstep++;
		#if(PRESTEP || PRESTEP2)
		for (n = 0; n < n_active; n++) {
			if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && (block[n_ord[n]][AMR_PRESTEP] != 0))block[n_ord[n]][AMR_PRESTEP] = 0;
			else if (block[n_ord[n]][AMR_PRESTEP] == 1)block[n_ord[n]][AMR_PRESTEP] = 2;
		}
		#endif
	}

	//fprintf(stderr, "\n h_dt(GPU%d): %f     ", rank, ndt);
	fprintf(stderr, "\n f_dt(GPU%d): %f     ", rank, ndt);

	//for (n = 0; n < n_active; n++){
		//ndt = advance(p, p, 0.5*dt, ph, 0, n_ord[n]);
		//fixup(ph, n_ord[n]);
		//bound_prim( n_ord[n]);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
		//fixup_utoprim(ph,n_ord[n]);  /* Fix the failure points using interpolation and updated ghost zone values */
		//bound_prim(ph,1,n_ord[n]);    /* Reset boundary conditions with fixed up points */
	//}

	// Danat: CPU evolve
	nstep = nstep - 2 * AMR_MAXTIMELEVEL;

	for (u = 0; u < 2 * AMR_MAXTIMELEVEL; u++) {
		set_prestep();
		ndt = advance(0);

		for (n = 0; n < n_active; n++) {
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  fixup(p, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup(ph, n_ord[n]);
		}

		bound_prim(ph, 0);    /* Set boundary conditions for primitive variables, flag bad ghost zones */

		nstep++;
	}

	//fprintf(stderr, "h_dt(CPU%d): %f \n ", rank, ndt);
	fprintf(stderr, "f_dt(CPU%d): %f \n ", rank, ndt);


	/*Temporary store p array from CPU to test array so ph array from GPU can be loaded*/
	for (n = 0; n < n_active; n++) {
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
			PLOOP{
				F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
				F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
			}
		}
	}

	/*Read p array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	code produce inconsistent output*/
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	for (n = 0; n < n_active; n++) {
		//ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) {
			PLOOP{
				if (p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] > 1.001 || p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] < 0.999) {
					// In low-density regions B-field is at ~machine precision, and relative error might be large
					if (k < 8 && p[n_ord[n]][index_3D(n_ord[n], i, j, z)][0] > 1e-6 * 1e-7) {
						fprintf(stderr, " i2:%d, j:%d, z: %d, k: %d, rank: %d, value1: %f, value2: %f, rho: %e  \n", i, j, z, k, rank,
							log(p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.), log(F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.),
							p[n_ord[n]][index_3D(n_ord[n], i, j, z)][0]);
					}
				}
			}
		}
	}

	/*Restore CPU version of p array*/
	for (n = 0; n < n_active; n++) {
		//ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G) { 
			PLOOP{
				p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
				ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
			}
		}
	}

	///*Temporary store ph array from CPU to test array so ph array from GPU can be loaded*/
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//			F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//		}
	//	}
	//}
	//
	///*Read ph array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	//code produce inconsistent output*/
	//for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			if (ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] > 1.001 || ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] < 0.999){
	//				if (k != 8){
	//					fprintf(stderr, " i1:%d, j:%d, z:%d, k: %d, rank:% d, value1: %f value2: %f  \n", i, j, z, k, rank,
	//						log(ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.), log(F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.));
	//				}
	//			}
	//		}
	//	}
	//}
	//
	///*Restore CPU version of ph array*/
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//			p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//		}
	//	}
	//}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif
	//ndt = advance_GPU(dt,1);   /* time step primitive variables to the half step */
	//GPU_fixup(1);
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(1);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//fprintf(stderr, "f_dt(GPU%d): %f     ", rank, ndt);

	//for (n = 0; n < n_active; n++){
		//ndt = advance(p, ph, dt, p, 1, n_ord[n]);
		//fixup(p, n_ord[n]);
		//bound_prim(n_ord[n]);
		//fixup_utoprim(p,n_ord[n]);
		//bound_prim(p,1,n_ord[n]);
	//}
	//fprintf(stderr, "f_dt(CPU%d): %f\n ", rank, ndt);

	///*Temporary store p array from CPU to test array so ph array from GPU can be loaded*/
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//			F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//		}
	//	}
	//}
	//
	///*Read p array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	//code produce inconsistent output*/
	//for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			if (p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]>1.001 || p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] / F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] < 0.999){
	//				if (k != 8){
	//					fprintf(stderr, " i2:%d, j:%d, z: %d, k: %d, rank: %d, value1: %f, value2: %f  \n", i, j, z, k, rank,
	//						log(p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.), log(F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] * F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k]) / log(10.));
	//				}
	//			}
	//		}
	//	}
	//}
	//
	///*Restore CPU version of p array*/
	//for (n = 0; n < n_active; n++){
	//	ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], BS_1 + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + BS_2 + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + BS_3 + N3G - 1) {
	//		PLOOP{
	//			p[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F1[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//			ph[n_ord[n]][index_3D(n_ord[n], i, j, z)][k] = F2[n_ord[n]][index_3D(n_ord[n], i, j, z)][k];
	//		}
	//	}
	//}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += (double)(AMR_MAXTIMELEVEL)*dt; // Danat: multiplied by (double)(AMR_MAXTIMELEVEL)

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif

	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel(0); // Danat: added this

	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */

	/* done! */
#endif
}
