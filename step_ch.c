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
#include "decs_MPI.h"
/** algorithmic choices **/


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

	if (rank == 0){
		fprintf(stderr, "h");
	}

	for (u = 0; u < AMR_MAXTIMELEVEL; u++){
		set_prestep();
		ndt = advance(0);
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  fixup(p, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup(ph, n_ord[n]);
		}
		/*for (n = 0; n < n_active; n++){
			if (pflag[n_ord[n]][index(n_ord[n], N1_GPU_offset[n_ord[n]] - N1G, N2_GPU_offset[n_ord[n]] - N2G, N3_GPU_offset[n_ord[n]] - N3G)] == 100){
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) fixup_utoprim(p, n_ord[n]);  //Fix the failure points using interpolation and updated ghost zone values
				else fixup_utoprim(ph, n_ord[n]);
				pflag[n_ord[n]][index(n_ord[n] ,N1_GPU_offset[n_ord[n]] - N1G, N2_GPU_offset[n_ord[n]] - N2G, N3_GPU_offset[n_ord[n]] - N3G)] = 0;
			}
		}*/
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
	//nstep++;

	/* increment time */
	t += (double)(AMR_MAXTIMELEVEL)*0.5*dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/
	#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	#endif

	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel();
	#if(TIMESTEP_JET)
	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel_jet();
	#endif

	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
	/* done! */
}

/*Set the timelevel for a jet which is domain decomposed in the second dimension*/
void set_timelevel_jet(void){
	int i, j, z, l, ni, nj, nz;
	int i2, j2, l2, z2;
	ni = NB_1;
	nj = NB_2;
	nz = NB_3;
	int min_j[NB_1];

	//Calculate the minimum timestep for one slice in R assuming REF_1==REF_2==0 and REF_3==1
	for (i = 0; i < ni; i++){
		min_j[i] = 10000;
		for (l = 0; l < N_LEVELS; l++){
			for (j = 0; j < nj; j++)for (z = 0; z < nz*pow(1 + REF_3, l); z++){
				if (block[AMR_coord_linear(l, i, j, z)][AMR_ACTIVE] == 1) min_j[i] = MY_MIN(block[AMR_coord_linear(l, i, j, z)][AMR_TIMELEVEL], min_j[i]);
			}
		}
		for (l = 0; l < N_LEVELS; l++){
			for (j = 0; j < nj; j++)for (z = 0; z < nz*pow(1 + REF_3, l); z++){
				if (block[AMR_coord_linear(l, i, j, z)][AMR_ACTIVE] == 1)block[AMR_coord_linear(l, i, j, z)][AMR_TIMELEVEL] = min_j[i];
			}
		}
	}
}

/*Calculate for every block the timestep. This function should be node independent*/
void set_timelevel(void){
	int n;
	int i, j, z, l, ni, nj, nz;
	int task;
	int min_j[NB_1];

	ni = NB_1;
	nj = NB_2;
	nz = NB_3;
	const int i_max = log(AMR_MAXTIMELEVEL) / log(2)+1;
	for (n = 0; n < n_active; n++){
		for (i = 0; i < i_max; i++){
			if (bdt[n_ord[n]][0] / dt > pow(2, i) && nstep > 0) block[n_ord[n]][AMR_TIMELEVEL] = pow(2, i);
		}
	}

	//First make sure all nodes have the same information regarding the timestep
	//Send for every block (l,i,j,z) to block (l2,i,j2,z2) on other nodes using non-blocking send
	for (n = 0; n < n_active_total;n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] == rank){
			for (task = 0; task < numtasks; task++){
				if (task!=rank){
					rc = MPI_Isend(&block[n_ord_total[n]][AMR_TIMELEVEL], 1, MPI_INT, task, n_ord_total[n] % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
					MPI_Request_free(&req[0]);
				}
			}
		}
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			rc = MPI_Irecv(&(block[n_ord_total[n]][AMR_TIMELEVEL]), 1, MPI_INT, block[n_ord_total[n]][AMR_NODE], n_ord_total[n] % MPI_TAG_MAX, mpi_cartcomm, &request_timelevel[n_ord_total[n]]);
		}
	}

	//Receive from other nodes using blocking receive
	for (n = 0; n < n_active_total; n++){
		if (block[n_ord_total[n]][AMR_ACTIVE] == 1 && block[n_ord_total[n]][AMR_NODE] != rank){
			MPI_Wait(&request_timelevel[n_ord_total[n]], &Statbound[n_ord[0]][0]);
		}
	}

	//Fixate the timestep around the pole
	for (i = 0; i < ni; i++){
		min_j[i] = 10000;
		if (block[AMR_coord_linear(0, i, 0, 0)][AMR_POLE] == 1 || block[AMR_coord_linear(0, i, 0, 0)][AMR_POLE] == 2 || block[AMR_coord_linear(0, i, 0, 0)][AMR_POLE] == 3){
			for (z = 0; z < NB_3; z++){
				min_j[i] = MY_MIN(block[AMR_coord_linear(0, i, 0, z)][AMR_TIMELEVEL], min_j[i]);
			}
			for (z = 0; z < NB_3; z++){
				block[AMR_coord_linear(0, i, 0, z)][AMR_TIMELEVEL] = min_j[i];
			}
		}
		min_j[i] = 10000;
		if (block[AMR_coord_linear(0, i, nj - 1, 0)][AMR_POLE] == 1 || block[AMR_coord_linear(0, i, nj - 1, 0)][AMR_POLE] == 2 || block[AMR_coord_linear(0, i, nj - 1, 0)][AMR_POLE] == 3){
			for (z = 0; z < NB_3; z++){
				min_j[i] = MY_MIN(block[AMR_coord_linear(0, i, nj - 1, z)][AMR_TIMELEVEL], min_j[i]);
			}
			for (z = 0; z < NB_3; z++){
				block[AMR_coord_linear(0, i, nj - 1, z)][AMR_TIMELEVEL] = min_j[i];
			}
		}
	}

	//Create communicators for nodes which have a minimum (i) timelevel
	int min_timelevel[8];
	for (i = 0; i <= log(AMR_MAXTIMELEVEL) / log(2); i++){
		if(nstep > 2 * AMR_SWITCHTIMELEVEL) MPI_Comm_free(&row_comm[i]);

		min_timelevel[i] = rank + 1000;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_TIMELEVEL] <= pow(2,i)) min_timelevel[i] = 1;
		}
		MPI_Comm_split(mpi_cartcomm, min_timelevel[i], rank, &row_comm[i]);
	}
	set_corners();
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
	int i, j, z, k, n;
	double ndt, U[NPR], dU[NPR];
	int ind0, ind1, ind2, ind3;

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#pragma omp parallel shared(n,p, ph, n_ord,N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset,nthreads) private(i,j,z,k)
			{
				#pragma omp for collapse(2) schedule(static,N1_GPU[n_ord[n]]*N2_GPU[n_ord[n]]/nthreads)
				ZLOOP3D_MPI{
					ind0 = index(n_ord[n], i, j, z);
					#pragma ivdep
					PLOOP ph[n_ord[n]][ind0][k] = p[n_ord[n]][ind0][k];        /* needed for Utoprim */
				}
			}
		}
	}

	ndt1 = ndt2 = ndt3 = 1e9;
	for (n = 0; n < n_active; n++){
		bdt[n_ord[n]][0] = bdt[n_ord[n]][1] = bdt[n_ord[n]][2] = bdt[n_ord[n]][3] = 1e9;
	}
	#if(N1G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[n_ord[n]][1] = fluxcalc(ph, F1, 1, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[n_ord[n]][1] = fluxcalc(p, F1, 1, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1]);
		}
		else if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
			ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send1(F1, Bufferp_1, n_ord[n]);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec1(F1, Bufferp_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec1(F1, Bufferp_1, n_ord[n], 2);

	#endif
	#if(N2G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[n_ord[n]][2] = fluxcalc(ph, F2, 2, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[n_ord[n]][2] = fluxcalc(p, F2, 2, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2]);
		}
		else if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
			ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send2(F2, Bufferp_1, n_ord[n]);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec2(F2, Bufferp_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec2(F2, Bufferp_1, n_ord[n], 2);

	#endif
	#if(N3G>0)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  bdt[n_ord[n]][3] = fluxcalc(ph, F3, 3, 1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bdt[n_ord[n]][3] = fluxcalc(p, F3, 3, 0, n_ord[n]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3]);
		}
		else if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
			ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_send3(F3, Bufferp_1, n_ord[n]);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec3(F3, Bufferp_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) flux_rec3(F3, Bufferp_1, n_ord[n], 2);
	#endif
	for (n = 0; n < n_active; n++) if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) fix_flux(F1, F2, F3, n_ord[n]);

	#if(!STAGGERED)
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1)  flux_ct(F1, F2, F3, n_ord[n]);
	#else
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  const_transport1(ph, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport1(p, n_ord[n]);
	}
	const_transport_bound();
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  const_transport2(ps, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) const_transport2(ps, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#endif
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)  utoprim(p, ph, p, ps, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) utoprim(p, p, ph, psh, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	ndt = 1e9;

	for (n = 0; n < n_active; n++){
		bdt[n_ord[n]][0] = 1. / (1. / bdt[n_ord[n]][1] + 1. / bdt[n_ord[n]][2] + 1. / bdt[n_ord[n]][3]);
		if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			ndt = MY_MIN(ndt, bdt[n_ord[n]][0]);
		}
		else if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
			ndt = MY_MIN(ndt, bdt[n_ord[n]][0] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
		}
	}
	

	//ndt=defcon*1./(1./ndt1+1./ndt2+1./ndt3);

	return defcon*ndt;
}

void utoprim(double(*restrict pi[NB])[NPR], double(*restrict pb[NB])[NPR], double(*restrict pf[NB])[NPR], double(*restrict psf[NB])[NDIM], double Dt, int n)
{
	int i, j, z, k;
	double ndt, ndt1, ndt2, ndt3, U[NPR], dU[NPR];
	struct of_geom geom;
	struct of_state q;
	int ind0, ind1, ind2, ind3;

	#pragma omp  parallel default(none) shared(n,gdet, pi,pb, pf, psf,stor1, stor2, dU_s, Katm, failimage, Dt, F1, F2,F3, pflag, dx, N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads, gam) private(i,j,z,k, geom, q, U, dU, ind0, ind1, ind2,ind3)
	{
		#pragma omp for collapse(2) schedule(static,N1_GPU[n]*N2_GPU[n]/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1){
			get_geometry(n, i, j, z, CENT, &geom);
			source(pb[n][index(n, i, j, z)], &geom, n, i, j, z, dU, Dt);
			get_state(pi[n][index(n, i, j, z)], &geom, &q);
			primtoU(pi[n][index(n, i, j, z)], &q, &geom, U);
			ind0 = index(n, i, j, z);
			ind1 = index(n, i + D1, j, z);
			ind2 = index(n, i, j + D2, z);
			ind3 = index(n, i, j, z + D3);
			#pragma ivdep
			PLOOP{
				U[k] += Dt*(
				#if( N1G > 0 )
				- (F1[n][ind1][k] - F1[n][ind0][k]) / dx[n][1]
				#endif
				#if( N2G > 0 )
				- (F2[n][ind2][k] - F2[n][ind0][k]) / dx[n][2]
				#endif
				#if( N3G > 0 )
				- (F3[n][ind3][k] - F3[n][ind0][k]) / dx[n][3]
				#endif
				+ dU[k]);
			}

			#if(ELLIPTICAL2)
			if(z==0){
				PLOOP U[k] += Dt*(dU_s[n][index2(n,i,j,z)][k]);
			}
			#endif

			#if STAGGERED
			U[B1] = 0.5*(psf[n][index(n, i, j, z)][1] * gdet[n][index2(n, i, j, z)][FACE1] + psf[n][index(n, i + D1, j, z)][1] * gdet[n][index2(n, i + D1, j, z)][FACE1]);
			U[B2] = 0.5*(psf[n][index(n, i, j, z)][2] * gdet[n][index2(n, i, j, z)][FACE2] + psf[n][index(n, i, j + D2, z)][2] * gdet[n][index2(n, i, j + D2, z)][FACE2]);
			#if(N3G>0)
			U[B3] = 0.5*(psf[n][index(n, i, j, z)][3] * gdet[n][index2(n, i, j, z)][FACE3] + psf[n][index(n, i, j, z + D3)][3] * gdet[n][index2(n, i, j, z + D3)][FACE3]);
			#endif
			#endif
			pflag[n][ind0] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf[n][ind0]);

			#if( DO_FONT_FIX ) 
			if (pflag[n][index(n, i, j, z)]) {
				failimage[n][index(n, i, j, z)][0]++;
				#if DOKTOT
				pflag[n][index(n, i, j, z)] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf[n][index(n, i, j, z)], pf[n][index(n, i, j, z)][KTOT]);
				#endif
				if (pflag[n][index(n, i, j, z)]) {
					failimage[n][index(n, i, j, z)][1]++;
					if (pflag[n][index(n, i, j, z)]){
						pflag[n][index(n, i, j, z)] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf[n][index(n, i, j, z)], pf[n][index(n, i, j, z)][KTOT]);
						pflag[n][index(n, N1_GPU_offset[n] - N1G, N2_GPU_offset[n] - N2G, N3_GPU_offset[n] - N3G)] = 100;
						failimage[n][index(n, i, j, z)][2]++;
					}
				}
			}
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
double fluxcalc(double(*restrict pr[NB])[NPR], double(*restrict F[NB])[NPR], int dir, int flag, int n)
{
	int i, j, z, k, idel, jdel, zdel, face;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR], F_HLL[NPR], U_HLL[NPR], vcon[NDIM], U_i[NPR], ptot;
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ndt, ndt_thread, dtij;
	double ctop;
	struct of_geom geom;
	struct of_state state_l, state_r, state_roe, qi;
	double bsq;
	int max_i, max_j, max_z;
	double val;
	ndt = 1.e9;
	int ind0, ind1;
	int fail_HLLC=0;
	int counter0 = 0;
	int counter1 = 0;
	double test;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }
	else { exit(10); }
	
		#pragma omp parallel default(none) shared(counter0,counter1,block, n_ord,n_active,n, gam, ps,t, psh,flag, pr, dq, ndt, cour, dx,dir,  F, face, idel, jdel, zdel, N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l, state_r, state_roe, F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ctop, dtij, ind0, ind1, U_HLL, F_HLL, qi, vcon, U_i, bsq, fail_HLLC, test, ptot)
		{
			ndt_thread = 1.e9;
			#if(RESCALE)
			/** evaluate slopes of primitive variables **/
			/* first rescale */
			#pragma omp for schedule(static,1)
			ZSLOOP(N1_GPU_offset[n] - 2, N1_GPU_offset[n] + N1_GPU[n] + 1, N2_GPU_offset[n] - 2, N2_GPU_offset[n] + N2_GPU[n] + 1) 	{
				get_geometry(n,i, j,z, CENT, &geom);
				rescale(pr[n][index(n ,i, j, z)], FORWARD, dir, i, j, CENT, &geom);
			}
			#endif

			/* then evaluate slopes */
			#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+2*D1)*(N2_GPU[n]+2*D2)/nthreads)
			ZSLOOP3D(N1_GPU_offset[n] - D1, N1_GPU_offset[n] + N1_GPU[n] - 1 + D1, N2_GPU_offset[n] - D2, N2_GPU_offset[n] + N2_GPU[n] - 1 + D2, N3_GPU_offset[n] - D3, N3_GPU_offset[n] + N3_GPU[n] - 1 + D3){
				// #pragma ivdep
				PLOOP{
					dq[n][index(n, i, j, z)][k] = slope_lim(pr[n][index(n, i - idel, j - jdel, z - zdel)][k], pr[n][index(n, i, j, z)][k], pr[n][index(n, i + idel, j + jdel, z + zdel)][k]);
				}
			}

			#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+jdel+zdel+1)*(N2_GPU[n]+idel+zdel+1)/nthreads)
			ZSLOOP((N1_GPU_offset[n] - jdel - zdel)*D1, (N1_GPU_offset[n] + N1_GPU[n])*D1, (N2_GPU_offset[n] - idel - zdel)*D2, (N2_GPU_offset[n] + N2_GPU[n])*D2) 	{
				for (z = (N3_GPU_offset[n] - idel - jdel)*D3; z <= (N3_GPU_offset[n] + N3_GPU[n])*D3; z++){
					get_geometry(n, i, j, z, face, &geom);
					/* this avoids problems on the pole */
					ind0 = index(n, i, j, z);
					ind1 = index(n, i - idel, j - jdel, z - zdel);

					#pragma ivdep
					PLOOP{
						p_l[k] = pr[n][ind1][k] + 0.5*dq[n][ind1][k];
						p_r[k] = pr[n][ind0][k] - 0.5*dq[n][ind0][k];
						#if(STAGGERED)
						if ((dir == 1 && k == B1)){
							if (flag == 0) p_l[k] = ps[n][ind0][k - (B1 - 1)];
							else p_l[k] = psh[n][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 2 && k == B2)){
							if (flag == 0) p_l[k] = ps[n][ind0][k - (B1 - 1)];
							else p_l[k] = psh[n][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 3 && k == B3)){
							if (flag == 0) p_l[k] = ps[n][ind0][k - (B1 - 1)];
							else p_l[k] = psh[n][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						#endif
					}

					#if(STAGGERED)
					if ((dir == 2) && ((j == 0 && (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3)) || (j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL])) && (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3)))){
						p_r[B1] = 0.;
						p_l[B1] = 0.;
					}
					#endif

					#if(RESCALE)
					get_geometry(n,i, j,z, CENT, &geom);
					rescale(p_l, REVERSE, dir, i, j, face, &geom);
					rescale(p_r, REVERSE, dir, i, j, face, &geom);
					#endif

					get_state(p_l, &geom, &state_l);
					get_state(p_r, &geom, &state_r);

					primtoflux(p_l, &state_l, dir, &geom, F_l);
					primtoflux(p_r, &state_r, dir, &geom, F_r);

					primtoflux(p_l, &state_l, TT, &geom, U_l);
					primtoflux(p_r, &state_r, TT, &geom, U_r);

					vchar(p_l, &state_l, &geom, dir, &cmax_l, &cmin_l, i, j, z);
					vchar(p_r, &state_r, &geom, dir, &cmax_r, &cmin_r, i, j, z);

					cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
					cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
					ctop = MY_MAX(cmax, cmin);

					#if(HLLC)
					if (dir == 1 && fabs(dot(state_r.bcon, state_r.bcov) / p_r[RHO]) > 0.0001 && fabs(dot(state_l.bcon, state_l.bcov) / p_r[RHO]) > 0.0001){
						//Get wavespeed defined as maximum of left and right state
						cmax_roe = MY_MAX(cmax_r, cmax_l);
						cmin_roe = MY_MIN(cmin_r, cmin_l);

						//Set U_HLL and F_HLL
						for (k = 0; k < NPR; k++){
							U_HLL[k] = (F_l[k] - F_r[k] + cmax_roe*U_r[k] - cmin_roe*U_l[k]) / (cmax_roe - cmin_roe + SMALL) / geom.g;
							F_HLL[k] = (cmax_roe*F_l[k] - cmin_roe*F_r[k] + cmax_roe*cmin_roe*(U_r[k] - U_l[k])) / (cmax_roe - cmin_roe + SMALL) / geom.g;
						}
						//note that the definition of the second component of the fluxes and conserved variables is a bit different, see gammie(2003)
						U_HLL[UU] -= U_HLL[RHO];
						F_HLL[UU] -= F_HLL[RHO];

						/*Set strength of magnetic field (free parameter in solution, will just take the average of the left and right
						extrapolated state for the moment untill we know better)*/
						//U_HLL[B1] = (p_l[B1] + p_r[B1]) / 2.;
						if (U_HLL[B1] == 0.0){
							fail_HLLC = 1;
							printf("error");
						}

						//If |B1|<0.01*|B2| || |B1|<0.01*|B3| revert to HLL flux
						if (fabs(U_HLL[B1] * sqrt(geom.gcov[1][1])) < 0.01*fabs(U_HLL[B2] * sqrt(geom.gcov[2][2])) || fabs(U_HLL[B1] * sqrt(geom.gcov[1][1])) < 0.01*fabs(U_HLL[B3] * sqrt(geom.gcov[3][3]))) fail_HLLC = 1;

						//Set initial guess 3-velocities of contact wave, take Roe average for the moment
						vcon[1] = 0.5*(sqrt(p_r[RHO])*state_r.ucon[1] / state_r.ucon[0] + sqrt(p_l[RHO])*state_l.ucon[1] / state_l.ucon[0]) / (sqrt(p_l[RHO]) + sqrt(p_r[RHO]));
						vcon[2] = 0.5*(sqrt(p_r[RHO])*state_r.ucon[2] / state_r.ucon[0] + sqrt(p_l[RHO])*state_l.ucon[2] / state_l.ucon[0]) / (sqrt(p_l[RHO]) + sqrt(p_r[RHO]));
						vcon[3] = 0.5*(sqrt(p_r[RHO])*state_r.ucon[3] / state_r.ucon[0] + sqrt(p_l[RHO])*state_l.ucon[3] / state_l.ucon[0]) / (sqrt(p_l[RHO]) + sqrt(p_r[RHO]));

						vcon[1] = state_r.ucon[1] / state_r.ucon[0];
						vcon[2] = state_r.ucon[2] / state_r.ucon[0];
						vcon[3] = state_r.ucon[3] / state_r.ucon[0];

						//Solve for velocities of intermediate state and calculate magnetic field b.
						solve_HLLC(&qi, &geom, vcon, U_HLL, F_HLL, &fail_HLLC);
						bsq = dot(qi.bcon, qi.bcov);
						//printf("testgamma: %f\n", dot(qi.ucon, qi.ucov));

						//if (vcon[1]<0.5*cmin_roe || vcon[1]>0.5*cmax_roe){
						//fail_HLLC = 1;
						//}
						if (fail_HLLC == 0){
							#pragma omp critical
							counter0++;
						}
						else{
							#pragma omp critical
							counter1++;
						}
						//Calculate total pressure ptot=pgas+0.5*bsq
						ptot = F_HLL[U1] + qi.bcon[1] * qi.bcov[1] - qi.ucov[1] / qi.ucov[0] * (F_HLL[UU] + qi.bcon[1] * qi.bcov[0]);
						if (cmax_roe>0. && qi.ucon[1] <= 0. && fail_HLLC == 0){
							PLOOP{
								U_r[k] /= geom.g;
								F_r[k] /= geom.g;
								U_l[k] /= geom.g;
								F_l[k] /= geom.g;
							}
							U_r[UU] -= U_r[RHO];
							F_r[UU] -= F_r[RHO];
							U_l[UU] -= U_l[RHO];
							F_l[UU] -= F_l[RHO];

							//Set Rankine-Hugoniot jump conditions
							U_i[RHO] = (cmax_roe - state_r.ucon[1] / state_r.ucon[0]) / (cmax_roe - vcon[1] + SMALL)*U_r[RHO];
							U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[1] - qi.bcon[1] * qi.bcov[0] - F_r[UU] + cmax_roe*U_r[UU]) / (cmax_roe - vcon[1]+SMALL);
							U_i[U1] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[1]/qi.ucov[0] - qi.bcon[0] * qi.bcov[1];
							U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[1] - qi.bcon[1] * qi.bcov[2] - F_r[U2] + cmax_roe*U_r[U2]) / (cmax_roe - vcon[1] + SMALL);
							U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[1] - qi.bcon[1] * qi.bcov[3] - F_r[U3] + cmax_roe*U_r[U3]) / (cmax_roe - vcon[1] + SMALL);
							U_i[B1] = U_HLL[B1];
							U_i[B2] = U_HLL[B2];
							U_i[B3] = U_HLL[B3];
							PLOOP{
								//test = 1. / geom.g*F[n][ind0][k] / (HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL)));
								//if (test > 1.25 || test<0.7){
								//}
								//printf("test %d: %f \n", k, fabs(cmin_roe / vcon[1]));

								//printf("test %d: %f %f \n", k, cmax_roe / vcon[1], 1. / geom.g*F[n][ind0][k] / (HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL))));
							}
							//Calculate HLLC flux
							PLOOP F[n][ind0][k] = geom.g*(F_r[k] + cmax_roe*(U_i[k] - U_r[k]));

							F[n][ind0][UU] += F[n][ind0][RHO];

						}
						else if (cmin_roe<0. && qi.ucon[1] >= 0. && fail_HLLC == 0){
							PLOOP{
								U_r[k] /= geom.g;
								F_r[k] /= geom.g;
								U_l[k] /= geom.g;
								F_l[k] /= geom.g;
							}
							U_r[UU] -= U_r[RHO];
							F_r[UU] -= F_r[RHO];
							U_l[UU] -= U_l[RHO];
							F_l[UU] -= F_l[RHO];

							//Set Rankine-Hugoniot jump conditions
							U_i[RHO] = (cmin_roe - state_l.ucon[1] / state_l.ucon[0]) / (cmin_roe - vcon[1] + SMALL)*U_l[RHO];
							U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[1] - qi.bcon[1] * qi.bcov[0] - F_l[UU] + cmin_roe*U_l[UU]) / (cmin_roe - vcon[1] + SMALL);
							U_i[U1] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[1];							
							U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[1] - qi.bcon[1] * qi.bcov[2] - F_l[U2] + cmin_roe*U_l[U2]) / (cmin_roe - vcon[1] + SMALL);
							U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[1] - qi.bcon[1] * qi.bcov[3] - F_l[U3] + cmin_roe*U_l[U3]) / (cmin_roe - vcon[1] + SMALL);
							U_i[B1] = U_HLL[B1];
							U_i[B2] = U_HLL[B2];
							U_i[B3] = U_HLL[B3];

							//Calculate HLLC flux
							PLOOP F[n][ind0][k] = geom.g*(F_l[k] + cmin_roe*(U_i[k] - U_l[k]));
							F[n][ind0][UU] += F[n][ind0][RHO];
						}
						else if (cmin_r >= 0. && fail_HLLC == 0){
							PLOOP F[n][ind0][k] = F_l[k];
						}
						else if (cmax_r <= 0. && fail_HLLC == 0){
							PLOOP F[n][ind0][k] = F_r[k];
						}
						else{ //revert to HLL flux
							PLOOP F[n][ind0][k] = HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
								+ LAXF*(0.5*(F_l[k] + F_r[k] - ctop*(U_r[k] - U_l[k])));
							fail_HLLC = 0;
						}

					}
					else{
						#pragma ivdep
						PLOOP F[n][ind0][k] = HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
							+ LAXF*(0.5*(F_l[k] + F_r[k] - ctop*(U_r[k] - U_l[k])));
					}
					#else
					#pragma ivdep
					PLOOP F[n][ind0][k] = HLLF*((cmax*F_l[k] + cmin*F_r[k] - cmax*cmin*(U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
						+ LAXF*(0.5*(F_l[k] + F_r[k] - ctop*(U_r[k] - U_l[k])));
					#endif

					/* evaluate restriction on timestep */
					cmax = MY_MAX(cmax, cmin);
					dtij = cour*dx[n][dir] / cmax;
					if (dtij < ndt_thread) {
						ndt_thread = dtij;
						#if(!TRANS_BOUND)
						if (dir == 2 && (j == 0 || j == N2 * pow(1+REF_2,block[n][AMR_LEVEL]))) {
							//#pragma ivdep
							PLOOP F[n][ind0][k] = 0.;
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

			#if(RESCALE)
			#pragma omp for schedule(static,1)
			ZSLOOP(N1_GPU_offset[n] - 2, N1_GPU_offset[n] + N1_GPU[n] + 1, N2_GPU_offset[n] - 2, N2_GPU_offset[n] + N2_GPU[n] + 1) 	{
				get_geometry(n,i, j,z, CENT, &geom);
				rescale(pr[n][index(n ,i, j, z)], REVERSE, dir, i, j, CENT, &geom);
			}
			#endif
		}
	return(ndt);
}
//Calculates function for which we try to find the root in solve_HLLC()
double func_HLLC(struct of_state *qi, double U_HLL[NPR], double F_HLL[NPR]){
	double result = (pow(qi->ucov[1] / qi->ucov[0], 2.)*(F_HLL[UU] + qi->bcon[1] * qi->bcov[0]) + qi->ucov[1] / qi->ucov[0] *
		(qi->bcon[0] * qi->bcov[0] - qi->bcon[1] * qi->bcov[1] + U_HLL[UU] - F_HLL[U1]) - qi->bcon[0] * qi->bcov[1] - U_HLL[U1]);
	return result;
}

void solve_HLLC(struct of_state *qi, struct of_geom *geom, double vcon[NDIM], double U_HLL[NPR], double F_HLL[NPR], int *fail_HLLC){
	int max_steps = 8; //make this no limiting factor, we may play with this later to enhance speed of code
	double rinit;
	int step = 0;
	double temp;
	double delta_v1;
	double eps = 0.00001; //The relative precision vcon[1] should be calculated to
	double derivative, f1, f2, vcon_old;
	double x1 = 0.9/sqrt(geom->gcov[1][1]);
	double x2 = -0.9/sqrt(geom->gcov[1][1]);

	//Solve for vcon[1] using secant method, which is more robust than Newton-Raphson
	do{
		//printf("test %d: %f %f %f \n",step, x1, f1,f2);

		//Calculate ucon, ucov, bcon, bcov
		if (step == 0){
			vcon[1] = x2;
			vcon[2] = (U_HLL[B2] * vcon[1] - F_HLL[B2]) / U_HLL[B1];
			vcon[3] = (U_HLL[B3] * vcon[1] - F_HLL[B3]) / U_HLL[B1];
			vcon_to_ucon(vcon, qi, geom);
			lower(qi->ucon, geom, qi->ucov);
			bcon_calc(U_HLL, qi->ucon, qi->ucov, qi->bcon);
			lower(qi->bcon, geom, qi->bcov);
			f2 = func_HLLC(qi, U_HLL, F_HLL);
		}

		vcon[1] = x1;
		vcon[2] = (U_HLL[B2] * vcon[1] - F_HLL[B2]) / U_HLL[B1];
		vcon[3] = (U_HLL[B3] * vcon[1] - F_HLL[B3]) / U_HLL[B1];
		vcon_to_ucon(vcon, qi, geom);
		lower(qi->ucon, geom, qi->ucov);
		bcon_calc(U_HLL, qi->ucon, qi->ucov, qi->bcon);
		lower(qi->bcon, geom, qi->bcov);
		f1 = func_HLLC(qi, U_HLL, F_HLL);
		
		temp = x1;
		x1 = (x2*f1 - x1*f2) / (f1 - f2);
		x2 = temp;
		f2 = f1;
		step++;
	} while (step<max_steps);
	vcon[1] = x1;
	vcon[2] = (U_HLL[B2] * vcon[1] - F_HLL[B2]) / U_HLL[B1];
	vcon[3] = (U_HLL[B3] * vcon[1] - F_HLL[B3]) / U_HLL[B1];
	vcon_to_ucon(vcon, qi, geom);
	lower(qi->ucon, geom, qi->ucov);
	bcon_calc(U_HLL, qi->ucon, qi->ucov, qi->bcon);
	lower(qi->bcon, geom, qi->bcov);
	if (x1 * sqrt(geom->gcov[1][1])>-0.05 && x1 * sqrt(geom->gcov[1][1]) < 0.05) fail_HLLC[0] = 0;
	else fail_HLLC[0] = 1;
	if (qi->ucon[1] * qi->ucov[1] + qi->ucon[2] * qi->ucov[2] + qi->ucon[3] * qi->ucov[3] >= 1. || fabs(f1)>0.001 || fabs(f2)>0.001){
		fail_HLLC[0] = 1;
	}
	//if (fail_HLLC[0] == 0) printf("test %d: %f %f %f \n", step, x1 * sqrt(geom->gcov[1][1]), f1, f2);
	//fail_HLLC[0] = 1;

}

/*Converts 3-velocity vcon[1-3] to 4-velocity ucon[0-3]*/
void vcon_to_ucon(double vcon[NDIM], struct of_state *q, struct of_geom *geom){
	double a;

	a = geom->gcov[0][0] + 2.*vcon[1] * geom->gcov[0][1] + 2.*vcon[2] * geom->gcov[0][2] + 2.*vcon[3] * geom->gcov[0][3]
		+ vcon[1] * vcon[1] * geom->gcov[1][1] + vcon[2] * vcon[2] * geom->gcov[2][2] + vcon[3] * vcon[3] * geom->gcov[3][3] + 
		2.*vcon[1] * vcon[2] * geom->gcov[1][2] + 2. * vcon[1] * vcon[3] * geom->gcov[1][3] + 2.*vcon[2] * vcon[3] * geom->gcov[2][3];

	q->ucon[0] = 1. / sqrt(fabs(a));
	q->ucon[1] = vcon[1] * q->ucon[0];
	q->ucon[2] = vcon[2] * q->ucon[0];
	q->ucon[3] = vcon[3] * q->ucon[0];
}
#if(STAGGERED)
void const_transport1(double(*restrict pb[NB])[NPR], int n){
	int i, j, z, k, ind0;
	double E_cent[NDIM];
	struct of_state q;
	struct of_geom geom;

#pragma omp parallel shared(n,n_ord,n_active,E_corn, F1, F2, F3, dx,pb,N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0, E_cent, geom, q)
	{
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1-D1, (N1_GPU_offset[n] + N1_GPU[n])*D1, N2_GPU_offset[n] * D2-D2, (N2_GPU_offset[n] + N2_GPU[n])*D2, N3_GPU_offset[n] * D3-D3, (N3_GPU_offset[n] + N3_GPU[n])*D3){
			ind0 = index(n, i, j, z);

			//calculate the corner values of the electric field by averaging the Godunov fluxes, see formula 7 balsara&spicer
			#if(N3G>0)
			E_corn[n][ind0][1] = 0.25*(F3[n][ind0][B2] + F3[n][index(n, i, j - D2, z)][B2] - F2[n][ind0][B3] - F2[n][index(n, i, j, z - D3)][B3]);
			E_corn[n][ind0][2] = 0.25*(F1[n][ind0][B3] + F1[n][index(n, i, j, z - D3)][B3] - F3[n][ind0][B1] - F3[n][index(n, i - D1, j, z)][B1]);
			#endif
			E_corn[n][ind0][3] = 0.25*(F2[n][ind0][B1] + F2[n][index(n, i - D1, j, z)][B1] - F1[n][ind0][B2] - F1[n][index(n, i, j - D2, z)][B2]);

			get_geometry(n, i, j, z, CENT, &geom);
			get_state(pb[n][ind0], &geom, &q);

			//calculate the cell center values of the E-field
			#if(N3G>0)
			E_cent[1] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
			E_cent[2] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
			#endif
			E_cent[3] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);

			//upwind the electric field based on transverse gradients conform gardiner&stone 2005/2015, not yet tested
			#if(N3G>0)
			dE[n][ind0][LEFT][1][2] = (E_cent[1] + F2[n][ind0][B3]);
			dE[n][ind0][LEFT][1][3] = (E_cent[1] - F3[n][ind0][B2]);
			dE[n][ind0][LEFT][2][1] = (E_cent[2] - F1[n][ind0][B3]);
			dE[n][ind0][LEFT][2][3] = (E_cent[2] + F3[n][ind0][B1]);
			#endif
			dE[n][ind0][LEFT][3][1] = (E_cent[3] + F1[n][ind0][B2]);
			dE[n][ind0][LEFT][3][2] = (E_cent[3] - F2[n][ind0][B1]);

			#if(N3G>0)
			dE[n][ind0][RIGHT][1][2] = (-F2[n][index(n, i, j + D2, z)][B3] - E_cent[1]);
			dE[n][ind0][RIGHT][1][3] = (F3[n][index(n, i, j, z + D3)][B2] - E_cent[1]);
			dE[n][ind0][RIGHT][2][1] = (F1[n][index(n, i + D1, j, z)][B3] - E_cent[2]);
			dE[n][ind0][RIGHT][2][3] = (-F3[n][index(n, i, j, z + D3)][B1] - E_cent[2]);
			#endif
			dE[n][ind0][RIGHT][3][1] = (-F1[n][index(n, i + D1, j, z)][B2] - E_cent[3]);
			dE[n][ind0][RIGHT][3][2] = (F2[n][index(n, i, j + D2, z)][B1] - E_cent[3]);
		}

		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1, (N1_GPU_offset[n] + N1_GPU[n])*D1, N2_GPU_offset[n] * D2, (N2_GPU_offset[n] + N2_GPU[n])*D2, N3_GPU_offset[n] * D3, (N3_GPU_offset[n] + N3_GPU[n])*D3){
			ind0 = index(n, i, j, z);
			double v[NDIM];
			get_geometry(n, i, j, z, CENT, &geom);
			ucon_calc(pb[n][index(n, i, j, z)], &geom, v);

			E_corn[n][ind0][1] = 0.25*((-F2[n][ind0][B3] - (dE[n][ind0][LEFT][1][3] * (double)(v[2]<=0.0) + dE[n][index(n, i, j - D2, z)][LEFT][1][3] * (double)(v[2]>0.0)))
				+ (-F2[n][index(n, i, j, z - D3)][B3] + (dE[n][index(n, i, j, z - D3)][RIGHT][1][3] * (double)(v[2]<=0.0) + dE[n][index(n, i, j - D2, z - D3)][RIGHT][1][3] * (double)(v[2]>0.0)))
				+ (F3[n][ind0][B2] - (dE[n][ind0][LEFT][1][2] * (double)(v[3]<=0.0) + dE[n][index(n, i, j, z - D3)][LEFT][1][2] * (double)(v[3]>0.0)))
				+ (F3[n][index(n, i, j - D2, z)][B2] + (dE[n][index(n, i, j - D2, z)][RIGHT][1][2] * (double)(v[3]<=0.0) + dE[n][index(n, i, j - D2, z - D3)][RIGHT][1][2] * (double)(v[3]>0.0))));
			E_corn[n][ind0][2] = 0.25*((-F3[n][ind0][B1] - (dE[n][ind0][LEFT][2][1] * (double)(v[3] <= 0.0) + dE[n][index(n, i, j, z - D3)][LEFT][2][1] * (double)(v[3] > 0.0)))
				+ (-F3[n][index(n, i - D1, j, z)][B1] + (dE[n][index(n, i - D1, j, z)][RIGHT][2][1] * (double)(v[3] <= 0.0) + dE[n][index(n, i - D1, j, z - D3)][RIGHT][2][1] * (double)(v[3] > 0.0)))
				+ (F1[n][ind0][B3] - (dE[n][ind0][LEFT][2][3] * (double)(v[1] <= 0.0) + dE[n][index(n, i - D1, j, z)][LEFT][2][3] * (double)(v[1] > 0.0)))
				+ (F1[n][index(n, i, j, z - D3)][B3] + (dE[n][index(n, i, j, z - D3)][RIGHT][2][3] * (double)(v[1] <= 0.0) + dE[n][index(n, i - D1, j, z - D3)][RIGHT][2][3] * (double)(v[1] > 0.0))));
			E_corn[n][ind0][3] = 0.25*((F2[n][ind0][B1] - (dE[n][ind0][LEFT][3][1] * (double)(v[2] <= 0.0) + dE[n][index(n, i, j - D2, z)][LEFT][3][1] * (double)(v[2] > 0.0)))
				+ (F2[n][index(n, i - D1, j, z)][B1] + (dE[n][index(n, i - D1, j, z)][RIGHT][3][1] * (double)(v[2] <= 0.0) + dE[n][index(n, i - D1, j - D2, z)][RIGHT][3][1] * (double)(v[2] > 0.0)))
				+ (-F1[n][ind0][B2] - (dE[n][ind0][LEFT][3][2] * (double)(v[1] <= 0.0) + dE[n][index(n, i - D1, j, z)][LEFT][3][2] * (double)(v[1] > 0.0)))
				+ (-F1[n][index(n, i, j - D2, z)][B2] + (dE[n][index(n, i, j - D2, z)][RIGHT][3][2] * (double)(v[1] <= 0.0) + dE[n][index(n, i - D1, j - D2, z)][RIGHT][3][2] * (double)(v[1] > 0.0))));

			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL]))) E_corn[n][ind0][1] = 0.5*(-F2[n][ind0][B3] - F2[n][index(n, i, j, z - D3)][B3]);
			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL]))) E_corn[n][ind0][3] = 0.5*(F2[n][ind0][B1] + F2[n][index(n, i - D1, j, z)][B1]);
		}
	}
}

void const_transport_bound(void){
	int n;
	gpu = 0;
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, Bufferdq_1, n_ord[n], 2);
	#endif
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send1(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, Bufferdq_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send2(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, Bufferdq_1, n_ord[n], 2);

	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send3(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, Bufferdq_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, Bufferdq_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 *block[n_ord[n]][AMR_TIMELEVEL]) ==2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 *block[n_ord[n]][AMR_TIMELEVEL]) == 2 *block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, Bufferdq_1, n_ord[n],1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, Bufferdq_1, n_ord[n],2);

	for (n = 0; n < n_active; n++)if (nstep % (2 *block[n_ord[n]][AMR_TIMELEVEL]) == 2 *block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_send_corn(E_corn, Bufferdq_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 *block[n_ord[n]][AMR_TIMELEVEL]) == 2 *block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, Bufferdq_1, n_ord[n],1);
	for (n = 0; n < n_active; n++)if (nstep % (2 *block[n_ord[n]][AMR_TIMELEVEL]) == 2 *block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, Bufferdq_1, n_ord[n],2);
	#endif	
	#endif

	#if(TRANS_BOUND)
	E_average();
	#endif
}

void const_transport2(double(*restrict psi[NB])[NDIM], double(*restrict psf[NB])[NDIM], double Dt, int n){
	int i, j, z, k, ind0;
	#pragma omp parallel shared(n,n_ord,n_active,E_corn, gdet,psi,psf, dx,Dt, p,N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0)
	{
		//update the staggered field components
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n], N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1){
			ind0 = index(n, i, j, z);
			psf[n][index(n, i, j, z)][1] = psi[n][index(n, i, j, z)][1] - Dt / dx[n][2] * (E_corn[n][index(n, i, j + D2, z)][3] - E_corn[n][ind0][3]) / gdet[n][index2(n, i, j, z)][FACE1];
			#if(N3G>0)
			psf[n][index(n, i, j, z)][1] += Dt / dx[n][3] * (E_corn[n][index(n, i, j, z + D3)][2] - E_corn[n][ind0][2]) / gdet[n][index2(n, i, j, z)][FACE1];
			#endif
		}

		//update the staggered field components
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n], N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1){
			ind0 = index(n, i, j, z);
			psf[n][index(n, i, j, z)][2] = psi[n][index(n, i, j, z)][2] + Dt / dx[n][1] * (E_corn[n][index(n, i + D1, j, z)][3] - E_corn[n][ind0][3]) / gdet[n][index2(n, i, j, z)][FACE2];
			#if(N3G>0)
			psf[n][index(n, i, j, z)][2] += -Dt / dx[n][3] * (E_corn[n][index(n, i, j, z + D3)][1] - E_corn[n][ind0][1]) / gdet[n][index2(n, i, j, z)][FACE2];
			#endif		
		}

		//update the staggered field components
		#if(N3G>0)
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], (N3_GPU_offset[n] + N3_GPU[n])*D3){
			ind0 = index(n, i, j, z);
			psf[n][index(n, i, j, z)][3] = psi[n][index(n, i, j, z)][3] - Dt / dx[n][1] * (E_corn[n][index(n, i + D1, j, z)][2] - E_corn[n][ind0][2]) / gdet[n][index2(n, i, j, z)][FACE3]
				+ Dt / dx[n][2] * (E_corn[n][index(n, i, j + D2, z)][1] - E_corn[n][ind0][1]) / gdet[n][index2(n, i, j, z)][FACE3];
		}
		#endif

	}
}

#define MAX_RANK 18000
void E_average(void){
	int n, n1, n2, i, j, z, k, ind0, z_max, number, u, send_tag1[MAX_RANK], send_tag2[MAX_RANK];

	if (numtasks >= MAX_RANK && rank == 0)fprintf(stderr, "Please increase MAX_RANK in E_average! \n");

	//Read in average value of E1 at pole for every block on node
	for (n = 0; n < n_active; n++) if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1){
		read_E_avg(E_avg, E_avg_x, E_avg_y, n_ord[n]);
	}

	//If block is not on node send the data to other node over MPI
	for (i = 0; i < NB_1; i++){
		//Which nodes have an active block around a slice in phi for a given i
		if (nstep % (block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag1[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, 0, z);
					if (block[number][AMR_NODE] == u) send_tag1[u] = 1;
				}
			}
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, 0, z);
				if (block[number][AMR_NODE] != rank && send_tag1[rank] == 1){
					E_avg[number][0]=(double(*))calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
					rc = MPI_Irecv(&(E_avg[number][0][0]), (BS_1 + 2 * N1G), MPI_DOUBLE, block[number][AMR_NODE], (100 * NB + number) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[number][490]);
				}
				if (block[number][AMR_NODE] == rank){
					for (u = 0; u < numtasks; u++){
						if (send_tag1[u] == 1 && u!=rank){
							rc = MPI_Isend(&(E_avg[number][0][0]), (BS_1 + 2 * N1G), MPI_DOUBLE, u, (100 * NB + number) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
							MPI_Request_free(&req[0]);
						}
					}
				}
			}
		}
		if (nstep % (block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag2[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, NB_2 - 1, z);
					if (block[number][AMR_NODE] == u) send_tag2[u] = 1;
				}
			}
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, NB_2 - 1, z);
				if (block[number][AMR_NODE] != rank && send_tag2[rank] == 1){
					E_avg[number][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G), sizeof(double));
					rc = MPI_Irecv(&(E_avg[number][1][0]), (BS_1 + 2 * N1G), MPI_DOUBLE, block[number][AMR_NODE], (101 * NB + number) % MPI_TAG_MAX, mpi_cartcomm, &boundreqs[number][491]);
				}
				if (block[number][AMR_NODE] == rank){
					for (u = 0; u < numtasks; u++){
						if (send_tag2[u] == 1 && u != rank){
							rc = MPI_Isend(&(E_avg[number][1][0]), (BS_1 + 2 * N1G), MPI_DOUBLE, u, (101 * NB + number) % MPI_TAG_MAX, mpi_cartcomm, &req[0]);
							MPI_Request_free(&req[0]);
						}
					}
				}
			}
		}
	}

	for (i = 0; i < NB_1; i++){
		//Which nodes have an active block around a slice in phi for a given i
		if (nstep % (block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag1[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, 0, z);
					if (block[number][AMR_NODE] == u) send_tag1[u] = 1;
				}
			}
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, 0, z);
				if (block[number][AMR_NODE] != rank && send_tag1[rank] == 1){
					MPI_Wait(&boundreqs[number][490], &Statbound[number][490]);
				}
			}
		}

		if (nstep % (block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag2[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, NB_2 - 1, z);
					if (block[number][AMR_NODE] == u) send_tag2[u] = 1;
				}
			}
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, NB_2 - 1, z);
				if (block[number][AMR_NODE] != rank && send_tag2[rank] == 1){
					MPI_Wait(&boundreqs[number][491], &Statbound[number][491]);
				}
			}
		}
	}

	//Average the first component of the E_field for both poles
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1){
		if (block[n_ord[n]][AMR_POLE] == 1 || block[n_ord[n]][AMR_POLE] == 3){
			z_max = NB_3*pow(1 + REF_3, block[n_ord[n]][AMR_LEVEL]);
			for (z = 0; z < z_max; z++){
				number=AMR_coord_linear(block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
				for (i = 0; i < BS_1 + N1G; i++){
					if (z == 0)E_avg_new[n_ord[n]][0][i] = E_avg[number][0][i] / ((double)z_max);
					else E_avg_new[n_ord[n]][0][i] += E_avg[number][0][i] / ((double)z_max);
				}
			}
		}

		if (block[n_ord[n]][AMR_POLE] == 2 || block[n_ord[n]][AMR_POLE] == 3){
			z_max = NB_3*pow(1 + REF_3, block[n_ord[n]][AMR_LEVEL]);
			for (z = 0; z < z_max; z++){
				number=AMR_coord_linear(block[n_ord[n]][AMR_LEVEL], block[n_ord[n]][AMR_COORD1], block[n_ord[n]][AMR_COORD2], z);
				for (i = 0; i < BS_1 + N1G; i++){
					if (z == 0)E_avg_new[n_ord[n]][1][i] = E_avg[number][1][i] / ((double)z_max);
					else E_avg_new[n_ord[n]][1][i] += E_avg[number][1][i] / ((double)z_max);
				}
			}
		}
	}


	for (i = 0; i < NB_1; i++){
		if (nstep % (block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, 0, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag1[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, 0, z);
					if (block[number][AMR_NODE] == u) send_tag1[u] = 1;
				}
			}
			//Which nodes have an active block around a slice in phi for a given i
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, 0, z);
				if (block[number][AMR_NODE] != rank && send_tag1[rank] == 1){
					free(E_avg[number][0]);
				}
			}
		}
		if (nstep % (block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL]) == block[AMR_coord_linear(0, i, NB_2 - 1, 0)][AMR_TIMELEVEL] - 1){
			for (u = 0; u < numtasks; u++){
				send_tag2[u] = 0;
				for (z = 0; z < NB_3; z++){
					number = AMR_coord_linear(0, i, NB_2 - 1, z);
					if (block[number][AMR_NODE] == u) send_tag2[u] = 1;
				}
			}
			for (z = 0; z < NB_3; z++){
				number = AMR_coord_linear(0, i, NB_2 - 1, z);
				if (block[number][AMR_NODE] != rank && send_tag2[rank] == 1){
					free(E_avg[number][1]);
				}
			}
		}
	}

	//Write average value of E1 at pole for every block on node
	for (n = 0; n < n_active; n++) if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1){
		write_E_avg(E_avg_new, E_avg_new_x, E_avg_new_y, n_ord[n]);
	}
}

void read_E_avg(double(*restrict E_avg[NB][2]), double(*restrict E_avg_x[NB][2]), double(*restrict E_avg_y[NB][2]), int n){
	int i,i1,i2, z, z1,z2, isize, zsize;
	//double ph;
	i1 = 0;
	i2 = N1_GPU[n] + N1G;
	z1 = 0;
	z2 = N3_GPU[n] + N3G;
	isize = (N1_GPU[n] + N1G);
	zsize = (N3_GPU[n] + N3G);
	
	if (gpu == 1)cudaSetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		pack_send2_E(n, block[n][AMR_NBR1], i1, i2, 0, D2, z1, z2, isize, zsize, send1_fine, E_corn, &(BufferE_1[n]), &(send1_fine[n]), NULL);
		if (gpu == 1){
			cudaStreamSynchronize(commandQueueGPU[n]);
		}
		for (i = i1; i < i2; i++){
			E_avg[n][0][i] = 0.;
			if (gpu == 1) for (z = z1; z < N3_GPU[n] + D3; z++){
				E_avg[n][0][i] += send1_fine[n][(i - i1)*zsize + (z - z1)];
			}
			else for (z = z1; z < N3_GPU[n] + D3; z++) E_avg[n][0][i] += send1_fine[n][2 * (i - i1)*zsize + 2 * (z - z1) + 0];
			E_avg[n][0][i] /= (double)(N3_GPU[n] + D3);
		}
	}
	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		pack_send2_E(n, block[n][AMR_NBR3], i1, i2, N2_GPU[n], N2_GPU[n] + D2, z1, z2, isize, zsize, send3_fine, E_corn, &(BufferE_1[n]), &(send3_fine[n]), NULL);
		if (gpu == 1){
			cudaStreamSynchronize(commandQueueGPU[n]);
		}
		for (i = i1; i < i2; i++){
			E_avg[n][1][i] = 0.;
			if (gpu == 1) for (z = z1; z < N3_GPU[n] + D3; z++){
				E_avg[n][1][i] += send3_fine[n][(i - i1)*zsize + (z - z1)];
			}
			else for (z = z1; z < N3_GPU[n] + D3; z++) E_avg[n][1][i] += send3_fine[n][2 * (i - i1)*zsize + 2 * (z - z1) + 0];
			E_avg[n][1][i] /= (double)(N3_GPU[n] + D3);
		}
	}
}

void write_E_avg(double(*restrict E_avg[NB][2]), double(*restrict E_avg_x[NB][2]), double(*restrict E_avg_y[NB][2]), int n){
	int i, i1, i2, z, z1, z2, isize, zsize;
	//double  ph1, ph2;
	i1 = 0;
	i2 = N1_GPU[n] + N1G;
	z1 = 0;
	z2 = N3_GPU[n] + N3G;
	isize = (N1_GPU[n] + N1G);
	zsize = (N3_GPU[n] + N3G);
	if (gpu == 1)cudaSetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		for (i = i1; i < i2; i++){	
			if (gpu == 1)for (z = z1; z < z2; z++){
				receive1_fine[n][(i - i1)*zsize + (z - z1)] = E_avg[n][0][i];
			}
			else for (z = z1; z < z2; z++){
				receive1_fine[n][2 * (i - i1)*zsize + 2 * (z - z1) + 0] = E_avg[n][0][i];
			}
		}
		unpack_receive2_E(n, n, n, i1, i2, 0, D2, z1, z2, isize, zsize, receive1_fine, NULL, NULL, E_corn, &(BufferE_1[n]), &(receive1_fine[n]), NULL, NULL, NULL, 4, 0, 0, 0, 0);
	}
	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		for (i = i1; i < i2; i++){
			if (gpu == 1)for (z = z1; z < z2; z++){
				receive3_fine[n][(i - i1)*zsize + (z - z1)] = E_avg[n][1][i];
			}
			else for (z = z1; z < z2; z++){
				receive3_fine[n][2 * (i - i1)*zsize + 2 * (z - z1) + 0] = E_avg[n][1][i];
			}
		}
		unpack_receive2_E(n, n, n, i1, i2, N2_GPU[n], N2_GPU[n] + D2, z1, z2, isize, zsize, receive3_fine, NULL, NULL, E_corn, &(BufferE_1[n]), &(receive3_fine[n]), NULL, NULL, NULL, 4, 0, 0, 0, 0);
	}
}

#endif

/***********************************************************************************************/
/***********************************************************************************************
flux_ct():
---------
-- performs the flux-averaging used to preserve the del.B = 0 constraint (see Toth 2000);
Note that we use in this new version of HARM dq instead of emf as temporary storage!

***********************************************************************************************/
void flux_ct(double(*restrict F1[NB])[NPR], double(*restrict F2[NB])[NPR], double(*restrict F3[NB])[NPR], int n)
{
	int i, j, z;
	int ind0;

	/* calculate EMFs */
	/* Toth approach: just average */
	#pragma omp parallel shared(n,dq, F1, F2, F3) private(i,j,z, ind0)
	{
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n]+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1 + D1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + D2, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + D3){
			ind0 = index(n ,i, j, z);
			#if (N2G>0 && N3G>0)
			dq[n][ind0][1] = 0.25*(F2[n][ind0][B3] + F2[n][index(n ,i, j, z - 1)][B3] - F3[n][ind0][B2] - F3[n][index(n ,i, j - 1, z)][B2]);
			#endif
			#if (N1G>0 && N3G>0)
			dq[n][ind0][2] = 0.25*(F3[n][ind0][B1] + F3[n][index(n ,i - 1, j, z)][B1] - F1[n][ind0][B3] - F1[n][index(n ,i, j, z - 1)][B3]);
			#endif
			#if (N1G>0 && N2G>0)
			dq[n][ind0][3] = 0.25*(F1[n][ind0][B2] + F1[n][index(n ,i, j - 1, z)][B2] - F2[n][ind0][B1] - F2[n][index(n ,i - 1, j, z)][B1]);
			#else
			dq[n][ind0][3] = 0.25*(F1[n][ind0][B2] + F1[n][index(n ,i, j - 1, z)][B2]);
			#endif
		}

		/* rewrite EMFs as fluxes, after Toth */
		#pragma omp for collapse(2) schedule(static,(N1_GPU[n]+D1)*(N2_GPU[n])/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1 + D1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) 	{
			ind0 = index(n ,i, j, z);
			#if (N1G>0)
			F1[n][ind0][B1] = 0.;
			#endif
			#if (N1G>0 && N2G>0)
			F1[n][ind0][B2] = 0.5*(dq[n][ind0][3] + dq[n][index(n ,i, j + 1, z)][3]);
			#endif
			#if (N1G>0 && N3G>0)
			F1[n][ind0][B3] = -0.5*(dq[n][ind0][2] + dq[n][index(n ,i, j, z + 1)][2]);
			#endif
		}

		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + D2, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) 	{
			ind0 = index(n ,i, j, z);
			#if (N1G>0 && N2G>0)		
			F2[n][ind0][B1] = -0.5*(dq[n][ind0][3] + dq[n][index(n ,i + 1, j, z)][3]);
			#endif
			#if (N2G>0 && N3G>0)
			F2[n][ind0][B3] = 0.5*(dq[n][ind0][1] + dq[n][index(n ,i, j, z + 1)][1]);
			#endif
			#if(N2G>0)
			F2[n][ind0][B2] = 0.;
			#endif
		}

		#pragma omp for collapse(2) schedule(static,(N1_GPU[n])*(N2_GPU[n])/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + D3) 	{
			ind0 = index(n ,i, j, z);
			#if (N1G>0 && N3G>0)
			F3[n][ind0][B1] = 0.5*(dq[n][ind0][2] + dq[n][index(n ,i + 1, j, z)][2]);
			#endif
			#if (N2G>0 && N3G>0)
			F3[n][ind0][B2] = -0.5*(dq[n][ind0][1] + dq[n][index(n ,i, j + 1, z)][1]);
			#endif
			#if(N3G>0)
			F3[n][ind0][B3] = 0.;
			#endif
		}
	}
}


/*Used for debugging. Compares output from CPU version to output from GPU version*/
void step_ch_debug()
{
#if (OpenCL_enable==1)
	double ndt, inmsg;
	int i, j, z, k, n;
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif
	//ndt = advance_GPU(0.5*dt, 0);   /* time step primitive variables to the half step */
	//GPU_fixup(0);
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(0);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(0,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	fprintf(stderr, "\n h_dt(GPU%d): %f     ", rank, ndt);
	
	for (n = 0; n < n_active; n++){
		//ndt = advance(p, p, 0.5*dt, ph, 0, n_ord[n]);
		//fixup(ph, n_ord[n]);
		//bound_prim( n_ord[n]);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
		//fixup_utoprim(ph,n_ord[n]);  /* Fix the failure points using interpolation and updated ghost zone values */
		//bound_prim(ph,1,n_ord[n]);    /* Reset boundary conditions with fixed up points */
	}
	fprintf(stderr, "h_dt(CPU%d): %f \n ", rank, ndt);

	/*Temporary store ph array from CPU to test array so ph array from GPU can be loaded*/
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
				F2[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = p[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
			}
		}
	}

	/*Read ph array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	code produce inconsistent output*/
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				if (ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k] / F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] > 1.001 || ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k] / F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] < 0.999){
					if (k != 8){
						fprintf(stderr, " i1:%d, j:%d, z:%d, k: %d, rank:% d, value1: %f value2: %f  \n", i, j, z, k, rank,
							log(ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k] * ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k]) / log(10.), log(F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] * F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k]) / log(10.));
					}
				}
			}
		}
	}

	/*Restore CPU version of ph array*/
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = F2[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
			}
		}
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	#endif
	//ndt = advance_GPU(dt,1);   /* time step primitive variables to the half step */
	//GPU_fixup(1);
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	//GPU_fixuputoprim(1);  /* Fix the failure points using interpolation and updated ghost zone values */
	//GPU_boundprim(1,1);    /* Set boundary conditions for primitive variables, flag bad ghost zones */
	fprintf(stderr, "f_dt(GPU%d): %f     ", rank, ndt);
	
	for (n = 0; n < n_active; n++){
		//ndt = advance(p, ph, dt, p, 1, n_ord[n]);
		//fixup(p, n_ord[n]);
		//bound_prim(n_ord[n]);
		//fixup_utoprim(p,n_ord[n]);
		//bound_prim(p,1,n_ord[n]);
	}
	fprintf(stderr, "f_dt(CPU%d): %f\n ", rank, ndt);
	
	/*Temporary store p array from CPU to test array so ph array from GPU can be loaded*/
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = p[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
				F2[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
			}
		}
	}

	/*Read p array from GPU and compare to CPU version. Print when difference becomes too big. If this occurs, the OpenCL and CPU versions of the
	code produce inconsistent output*/
	for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				if (p[n_ord[n]][index(n_ord[n] ,i, j, z)][k] / F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k]>1.001 || p[n_ord[n]][index(n_ord[n] ,i, j, z)][k] / F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] < 0.999){
					if (k != 8){
						fprintf(stderr, " i2:%d, j:%d, z: %d, k: %d, rank: %d, value1: %f, value2: %f  \n", i, j, z, k, rank,
							log(p[n_ord[n]][index(n_ord[n] ,i, j, z)][k] * p[n_ord[n]][index(n_ord[n] ,i, j, z)][k]) / log(10.), log(F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k] * F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k]) / log(10.));
					}
				}
			}
		}
	}

	/*Restore CPU version of p array*/
	for (n = 0; n < n_active; n++){
		ZSLOOP3D(-2 + N1_GPU_offset[n_ord[n]], N1_GPU[n_ord[n]] + N1_GPU_offset[n_ord[n]] + 1, -2 + N2_GPU_offset[n_ord[n]], N2_GPU_offset[n_ord[n]] + N2_GPU[n_ord[n]] + 1, -N3G + N3_GPU_offset[n_ord[n]], N3_GPU_offset[n_ord[n]] + N3_GPU[n_ord[n]] + N3G - 1) {
			PLOOP{
				p[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = F1[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
				ph[n_ord[n]][index(n_ord[n] ,i, j, z)][k] = F2[n_ord[n]][index(n_ord[n] ,i, j, z)][k];
			}
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
	#if (MPI_enable)
	MPI_Barrier(mpi_cartcomm);
	MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	MPI_Barrier(mpi_cartcomm);
	#endif
	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
	
	/* done! */
#endif
}

void GPU_step_ch()
{
	double ndt, inmsg;
	int i, j, z, k, n, uu;

	if (rank == 0){
		//fprintf(stderr, "h");
	}
	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
	}
	for (uu = 0; uu < 2 * AMR_MAXTIMELEVEL; uu++){
		set_prestep();
		ndt = advance_GPU();   /* time step primitive variables to the half step */

		GPU_boundprim(0);    /* Set boundary conditions for primitive variables, flag bad ghost zones */

		nstep++;
		#if(PRESTEP)
		for (n = 0; n < n_active; n++){
			if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && (block[n_ord[n]][AMR_PRESTEP] != 0))block[n_ord[n]][AMR_PRESTEP] = 0;
			else if (block[n_ord[n]][AMR_PRESTEP] == 1)block[n_ord[n]][AMR_PRESTEP] = 2;
		}
		#endif
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	if (rank == 0){
		//fprintf(stderr, "f");
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += (double)(AMR_MAXTIMELEVEL)*dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/

	#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
	#endif

	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel();

	#if(TIMESTEP_JET)
	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0)set_timelevel_jet();
	#endif

	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
}

void set_prestep(void){
	int n;

	#if(PRESTEP)
	int timelevel_min = AMR_MAXTIMELEVEL;
	int blocks_per_timestep = 0;
	int blocks_this_timestep = 0;

	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	//Find the minimum timelevel on this node
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_TIMELEVEL] < timelevel_min) timelevel_min = block[n_ord[n]][AMR_TIMELEVEL];
	}

	//Calculate the number of blocks you want to evolve simultaneously
	blocks_per_timestep = (count_node[0] - count_node[0] % (AMR_MAXTIMELEVEL / timelevel_min)) / (AMR_MAXTIMELEVEL / timelevel_min);
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep++;
	}

	//If you don't have sufficient blocks this timestep preevolve some blocks if available
	if ((nstep % timelevel_min) == timelevel_min - 1){
		for (n = 0; n < n_active; n++){
			if (blocks_this_timestep < blocks_per_timestep && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) != block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0 && (block[n_ord[n]][AMR_POLE] == 0)){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
				blocks_this_timestep++;
			}
		}
	}

	#if(N_GPU>1)
	for (gpu = 0; gpu < N_GPU; gpu++){
		timelevel_min = AMR_MAXTIMELEVEL;
		blocks_per_timestep = 0;
		blocks_this_timestep = 0;
		//Find the minimum timelevel on this gpu
		for (n = 0; n < n_active_gpu[gpu]; n++){
			if (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] < timelevel_min) timelevel_min = block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL];
		}

		//Calculate the number of blocks you want to evolve simultaneously
		blocks_per_timestep = (count_gpu[gpu] - count_gpu[gpu] % (AMR_MAXTIMELEVEL / timelevel_min)) / (AMR_MAXTIMELEVEL / timelevel_min);
		for (n = 0; n < n_active_gpu[gpu]; n++){
			if (nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) == block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1 || block[n_ord[n]][AMR_PRESTEP] == 1 || block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep++;
		}

		//If you don't have sufficient blocks this timestep preevolve some blocks if available
		if ((nstep % timelevel_min) == timelevel_min - 1){
			for (n = 0; n < n_active_gpu[gpu]; n++){
				if (blocks_this_timestep < blocks_per_timestep && nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) != block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1
					&& (block[n_ord_gpu[gpu][n]][AMR_PRESTEP] == 0) && (block[n_ord_gpu[gpu][n]][AMR_POLE] == 0)){
					block[n_ord_gpu[gpu][n]][AMR_PRESTEP] = 1;
					block[n_ord_gpu[gpu][n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) - (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1));
					blocks_this_timestep++;
				}
			}
		}
	}
	#endif
	//If at end of switchtimelevel do not pre-evolve
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) >= 2 * AMR_SWITCHTIMELEVEL - 2 * AMR_MAXTIMELEVEL){
			block[n_ord[n]][AMR_PRESTEP] = 0;
			block[n_ord[n]][AMR_NSTEP] = nstep;
		}
	}
	#else
	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	#endif
}

double advance_GPU(void)
{
	int i, n;
	double timestep;
	gpu = 1;


	if (nstep % (2 * AMR_MAXTIMELEVEL) == 0){
		ndt1 = ndt2 = ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[n_ord[n]][0] = bdt[n_ord[n]][1] = bdt[n_ord[n]][2] = bdt[n_ord[n]][3] = 1e9;
		}
	}
	for (n = 0; n < n_active; n++){
		prestep_half[n_ord[n]] = (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) < block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
		prestep_full[n_ord[n]] = (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) < 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) >  block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
	}

	#if(N1G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(1, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(1, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][1] = fluxcalc_GPU(n_ord[n],1);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt1 = 1e9;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1]);
			}
			else{
				ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt1 = 1e9;
	#endif
	#if(N2G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(2, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(2, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][2] = fluxcalc_GPU(n_ord[n],2);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt2 = 1e9;
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2]);
			}
			else{
				ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt2 = 1e9;
	#endif
	#if(N3G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(3, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(3, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][3] = fluxcalc_GPU(n_ord[n],3);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3]);
			}
			else{
				ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt3 = 1e9;
	#endif

	gpu = 1;
	rc = 0;

	//MPI communication
	/*for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
	if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
	if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
	break;
	}
	}*/

	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		flux_send1(F1, BufferF1_1, n_ord[n]);
		flux_send2(F2, BufferF2_1, n_ord[n]);
		#if(N3G>0)
		flux_send3(F3, BufferF3_1, n_ord[n]);
		#endif
	}

	#if(PRESTEP)
	//For last timestep synchronize electric fields immediately
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		flux_rec1(F1, BufferF1_1, n_ord[n], 5);
		flux_rec2(F2, BufferF2_1, n_ord[n], 5);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 5);
		#endif
	}

	//For first timestep do not synchronize electrice fields 
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		flux_rec1(F1, BufferF1_1, n_ord[n], 2);
		flux_rec2(F2, BufferF2_1, n_ord[n], 2);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 2);
		#endif
	}
	#else
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		flux_rec1(F1, BufferF1_1, n_ord[n], 1);
		flux_rec2(F2, BufferF2_1, n_ord[n], 1);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 1);
		#endif
	}

	//For first timestep do not synchronize electrice fields
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)){ //
		flux_rec1(F1, BufferF1_1, n_ord[n], 2);
		flux_rec2(F2, BufferF2_1, n_ord[n], 2);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 2);
		#endif
	}
	#endif 
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomF \n");
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1 || prestep_half[n_ord[n]] == 1) GPU_fix_flux(n_ord[n]);
	#endif
	#if(STAGGERED)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport1(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport1(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport2(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport2(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
	rc = 0;
	GPU_consttransport_bound();
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE \n");
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport3(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport3(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}

	#else
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct1(n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct2(n_ord[n]);
	#endif

	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1){
			timestep = dt*(double)block[n_ord[n]][AMR_TIMELEVEL];
			GPU_fixup(1, n_ord[n], timestep);
		}
		else if (prestep_half[n_ord[n]] == 1){
			timestep = 0.5 * dt*(double)block[n_ord[n]][AMR_TIMELEVEL];
			GPU_fixup(0, n_ord[n], timestep);
		}
	}

	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[n_ord[n]][0] = 1. / (1. / bdt[n_ord[n]][1] + 1. / bdt[n_ord[n]][2] + 1. / bdt[n_ord[n]][3]);
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt = MY_MIN(ndt, bdt[n_ord[n]][0]);
			}
			else{
				ndt = MY_MIN(ndt, bdt[n_ord[n]][0] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}

	//ndt = defcon * 1. / (1. / ndt1 + 1. / ndt2 + 1. / ndt3);
	return defcon * ndt;
	return 0.;
}