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
//new
#include "decs_MPI.h"
void bound_prim1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);

/* bound array containing entire set of primitive variables */
void bound_prim(double(*restrict prim[NB_LOCAL])[NPR], int bound_force)
{
	int i, n, flag;
	double temp=nstep;
	if (bound_force == 1) nstep = -1;
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim1(p,ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim1(ph, psh, n_ord[n]);
	}

	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2(p,ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2(ph,psh, n_ord[n]);
	}
	#endif

	MPI_Barrier(MPI_COMM_WORLD);
	rc = 0;
	gpu = 0;
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	if (N3 > 1){
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		set_iprobe(0, &flag);
		do {
			for (n = 0; n < n_active; n++){
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
			}
			set_iprobe(1, &flag);
		} while (flag);
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");

	#if(TRANS_BOUND && NB_3==1)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim_trans(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim_trans(ph, psh, n_ord[n]);
	}
	#endif

	#if (STAGGERED && COPY_BFIELD)
	rc = 0;
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1 || bound_force == 1){ //watch out does this for both half and full timestep while only needed for full timestep
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send2(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send3(ps, Bufferps_1, n_ord[n]);
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) B_rec3(ps, Bufferps_1, n_ord[n]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB \n");
	#endif

	if (bound_force == 1) nstep = temp;
}

void set_iprobe(int mode, int * flag){
	int i ,n;
	*flag = 0;
	for (n = 0; n < n_active; n++){
		if (mode == 0){
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++) block[n_ord[n]][i] = 0;
		}
		else{
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++){
				if (block[n_ord[n]][i] == -1){
					block[n_ord[n]][i] = 0;
					*flag = 1;
				}
				else  block[n_ord[n]][i] = 1;
			}
		}
	}
	return;
}

void bound_prim1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	if (block[n][AMR_NBR4] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n]-N2G; j < N2_GPU_offset[n] + BS_2+N2G; j++){
				for (z = N3_GPU_offset[n]-N3G; z < N3_GPU_offset[n] + BS_3+N3G; z++){
					#if( RESCALE )
					get_geometry(0, j, CENT, &geom);
					rescale(prim[0][j], FORWARD, 1, 0, j, CENT, &geom);
					#endif
					//#pragma omp   simd
					for (i = -N1G; i < 0; i++){
						for (k = 0; k < NPR; k++){
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, 0, j, z)][k];
						}
						#if(STAGGERED)
						for (k = 2; k < NDIM; k++){
							ps[nl[n]][index_3D(n, i, j, z)][k] = ps[nl[n]][index_3D(n, 0, j, z)][k];
						}
						#endif
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, 0, j, z)];
					}
					#if( RESCALE )
					get_geometry(0, j, CENT, &geom);
					rescale(prim[0][j], REVERSE, 1, 0, j, CENT, &geom);
					get_geometry(-1, j, CENT, &geom);
					rescale(prim[-1][j], REVERSE, 1, -1, j, CENT, &geom);
					get_geometry(-2, j, CENT, &geom);
					rescale(prim[-2][j], REVERSE, 1, -2, j, CENT, &geom);
					#endif
				}
			}
		}
	}

	if (block[n][AMR_NBR2] == -1){
		// outer r BC: outflow 		
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++){
					#if( RESCALE )
					get_geometry(N1 - 1, j, CENT, &geom);
					rescale(prim[N1 - 1][j], FORWARD, 1, N1 - 1, j, CENT, &geom);
					#endif
					
					for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G; i++){
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)];
						#if(STAGGERED)
						for (k = 2; k < NDIM; k++){
							ps[nl[n]][index_3D(n, i, j, z)][k] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
						}
						#endif
					}
					#if( RESCALE )
					get_geometry(N1 - 1, j, CENT, &geom);
					rescale(prim[N1 - 1][j], REVERSE, 1, N1 - 1, j, CENT, &geom);
					get_geometry(N1, j, CENT, &geom);
					rescale(prim[N1][j], REVERSE, 1, N1, j, CENT, &geom);
					get_geometry(N1 + 1, j, CENT, &geom);
					rescale(prim[N1 + 1][j], REVERSE, 1, N1 + 1, j, CENT, &geom);
					#endif
				}
			}
		}
	}

	// make sure there is no inflow at the inner boundary 
	if (block[n][AMR_NBR4] == -1){
		for (i = -N1G; i <= -1; i++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
				#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, -1, j, z)], n, i, j, z, 0);
						inflow_check(prim[nl[n]][index_3D(n, -2, j, z)], n, i, j, z, 0);
						#if(N1G==3)
						inflow_check(prim[nl[n]][index_3D(n, -3, j, z)], n, i, j, z, 0);
						#endif
					}
				}
			}
		}
	}
	// make sure there is no inflow at the outer boundary
	if (block[n][AMR_NBR2] == -1){
		for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i <= N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G - 1; i++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
				#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]), j, z)], n, i, j, z, 1);
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 1, j, z)], n, i, j, z, 1);
						#if(N1G==3)
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 2, j, z)], n, i, j, z, 1);
						#endif
					}
				}
			}
		}
	}
}

void bound_prim2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k, jref;

	//copy all densities and B^phi in; interpolate linearly transverse velocity
	#if(POLEFIX && POLEFIX < N2/2)
	jref = POLEFIX;
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2 || (N3 > 1 && k == B3))
							//don't touch magnetic fields
							continue;
							else if (k == U2) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, jref, z)][k];
							}
							else {
								//everything else copy (both poles)
								prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, jref, z)][k];
							}
						}
					}
				}
			}
		}
	}
	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2 || (N3 > 1 && k == B3))
							//don't touch magnetic fields
							continue;
							else if (k == U2) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
							else {
								//everything else copy (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
						}
					}
				}
			}
		}
	}
	#endif

	// polar BCs 
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					PLOOP{
						prim[nl[n]][index_3D(n, i, -1, z)][k] = prim[nl[n]][index_3D(n, i, 0, z)][k];
						prim[nl[n]][index_3D(n, i, -2, z)][k] = prim[nl[n]][index_3D(n, i, 1, z)][k];
						#if(N1G==3)
						prim[nl[n]][index_3D(n, i, -3, z)][k] = prim[nl[n]][index_3D(n, i, 2, z)][k];
						#endif
					}
					pflag[nl[n]][index_3D(n, i, -1, z)] = pflag[nl[n]][index_3D(n, i, 0, z)];
					#if(STAGGERED)
					k = 1;
					ps[nl[n]][index_3D(n, i, -1, z)][k] = ps[nl[n]][index_3D(n, i, 0, z)][k];
					ps[nl[n]][index_3D(n, i, -2, z)][k] = ps[nl[n]][index_3D(n, i, 1, z)][k];
					#if(N2G==3)
					ps[nl[n]][index_3D(n, i, -3, z)][k] = ps[nl[n]][index_3D(n, i, 2, z)][k];
					#endif
					#if(N3>1)
					k = 3;
					ps[nl[n]][index_3D(n, i, -1, z)][k] = ps[nl[n]][index_3D(n, i, 0, z)][k];
					ps[nl[n]][index_3D(n, i, -2, z)][k] = ps[nl[n]][index_3D(n, i, 1, z)][k];
					#if(N2G==3)
					ps[nl[n]][index_3D(n, i, -3, z)][k] = ps[nl[n]][index_3D(n, i, 2, z)][k];
					#endif
					#endif			
					#endif
				}
			}
		}
	}

	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					PLOOP{
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
						#if(N1G==3)
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
						#endif
					}
					pflag[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)] = pflag[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)];
					#if(STAGGERED)
					k = 1;
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
					#if(N2G==3)
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
					#endif
					#if(N3>1)
					k = 3;
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
					#if(N2G==3)
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
					#endif
					#endif			
					#endif
				}
			}
		}
	}

	// make sure b and u are antisymmetric at the poles 
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
}


void bound_prim_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;

	// polar BCs 
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++){
						//#pragma omp   simd
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][k];
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B3] *= -1.0;

						#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][1];
						#if(N3>1)
						ps[nl[n]][index_3D(n, i, j, z)][3] = -ps[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][3];
						#endif			
						#endif
					}
				}
			}
		}
	}

	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++){
						//#pragma omp   simd
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1 , (z + BS_3 / 2) % BS_3)][k];
						prim[nl[n]][index_3D(n, i, j , z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B3] *= -1.0;

						#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1, (z + BS_3 / 2) % BS_3)][1];
						#if(N3>1)
						ps[nl[n]][index_3D(n, i, j, z)][3] = -ps[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1, (z + BS_3 / 2) % BS_3)][3];
						#endif			
						#endif
					}
				}
			}
		}
	}
}

void inflow_check(double * restrict pr, int n, int ii, int jj, int zz, int type){
    struct of_geom geom ;
    double ucon[NDIM] ;
    int j,k ;
    double alpha,beta1,gamma,vsq ;

    get_geometry(n, ii,jj,zz,CENT,&geom) ;
    ucon_calc(pr, &geom, ucon) ;

    if( ((ucon[1] > 0.) && (type==0)) || ((ucon[1] < 0.) && (type==1)) ) { 
		/* find gamma and remove it from primitives */
		if( gamma_calc(pr,&geom,&gamma) ) { 
			fprintf(stderr,"\ninflow_check(): gamma failure \n");
			fail(FAIL_GAMMA);
		}
		pr[U1] /= gamma ;
		pr[U2] /= gamma ;
		pr[U3] /= gamma ;
		alpha = 1./sqrt(-geom.gcon[0][0]) ;
		beta1 = geom.gcon[0][1]*alpha*alpha ;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[U1] = beta1/alpha ;

		/* now find new gamma and put it back in */
		vsq = 0. ;
		SLOOP vsq += geom.gcov[j][k]*pr[U1+j-1]*pr[U1+k-1] ;
		if( fabs(vsq) < 1.e-13 )  vsq = 1.e-13;
		if( vsq >= 1. ) { 
			vsq = 1. - 1./(GAMMAMAX*GAMMAMAX) ;
		}
		gamma = 1./sqrt(1. - vsq) ;
		pr[U1] *= gamma ;
		pr[U2] *= gamma ;
		pr[U3] *= gamma ;

		/* done */
	}
	else{
		return;
	}
}

