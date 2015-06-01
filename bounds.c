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


#include "decs.h"

/* bound array containing entire set of primitive variables */
void bound_prim(double (*prim)[NPR], int MPI)
{
    int i,j,z,k,jref, z1, z2 ;
	double trel;
    struct of_geom geom ;
	MPI_Barrier(MPI_COMM_WORLD);

    // inner r boundary condition: u, gdet extrapolation
	if (N1_MPI_offset == 0){
		#pragma omp parallel shared(prim, pflag,gdet, geom) private(i,j,z,k)
		{
			#pragma omp for schedule(static,1)	
			for (j = N2_MPI_offset; j < N2_MPI_offset + N2_MPI; j++){
				for (z = N3_MPI_offset-N3G; z < N3_MPI_offset + N3_MPI+N3G; z++){
					#if( RESCALE )
					get_geometry(0,j,CENT,&geom) ;
					rescale(prim[0][j],FORWARD, 1, 0,j,CENT,&geom) ;
					#endif
					#if(!boundfreeze1 && !boundfreeze2)
					//#pragma omp simd
					PLOOP prim[index(-1, j, z)][k] = prim[index(0, j, z)][k] * gdet[index2(0, j)][3] / gdet[index2(-1, j)][3];
					//#pragma omp simd
					PLOOP prim[index(-2, j, z)][k] = prim[index(0, j, z)][k] * gdet[index2(0, j)][3] / gdet[index2(-2, j)][3];
					pflag[index(-1, j, z)] = pflag[index(0, j, z)];
					//pflag[index(-2, j, z)] = pflag[index(0, j, z)];
					#elif(boundfreeze2)
					PLOOP prim[index(ibound, j, z)][k] = pbound[j*(N3_MPI+2*N3G)+z][k][0];
					PLOOP prim[index(ibound+1, j, z)][k] = pbound[j*(N3_MPI+2*N3G)+z][k][1];
					pflag[index(ibound, j, z)] = 0;
					pflag[index(ibound+1, j, z)] = 0;
					#endif
					#if( RESCALE )
					get_geometry(0,j,CENT,&geom) ;
					rescale(prim[0][j],REVERSE, 1, 0,j,CENT,&geom) ;
					get_geometry(-1,j,CENT,&geom) ;
					rescale(prim[-1][j],REVERSE, 1, -1,j,CENT,&geom) ;
					get_geometry(-2,j,CENT,&geom) ;
					rescale(prim[-2][j],REVERSE, 1, -2,j,CENT,&geom) ;
					#endif
				}
			}
		}
	}

    // outer r BC: outflow 		
	if (N1_MPI_offset + N1_MPI == N1){
		#pragma omp parallel shared(prim, pflag) private(i,j,k,z, geom)
		{
			#pragma omp for schedule(static,1)
				
			for (j = N2_MPI_offset; j < N2_MPI_offset + N2_MPI; j++){
				for (z = N3_MPI_offset-N3G; z < N3_MPI_offset + N3_MPI+N3G; z++){
					#if( RESCALE )
					get_geometry(N1-1,j,CENT,&geom) ;
					rescale(prim[N1-1][j],FORWARD, 1, N1-1,j,CENT,&geom) ;
					#endif
					//#pragma omp simd
					PLOOP prim[index(N1, j, z)][k] = prim[index(N1 - 1, j, z)][k];
					//#pragma omp simd
					PLOOP prim[index(N1 + 1, j, z)][k] = prim[index(N1 - 1, j, z)][k];
					pflag[index(N1, j, z)] = pflag[index(N1 - 1, j, z)];
					//pflag[index(N1 + 1, j, z)] = pflag[index(N1 - 1, j, z)];

					#if( RESCALE )
					get_geometry(N1-1,j,CENT,&geom) ;
					rescale(prim[N1-1][j],REVERSE, 1, N1-1,j,CENT,&geom) ;
					get_geometry(N1,j,CENT,&geom) ;
					rescale(prim[N1][j],REVERSE, 1, N1,j,CENT,&geom) ;
					get_geometry(N1+1,j,CENT,&geom) ;
					rescale(prim[N1+1][j],REVERSE, 1, N1+1,j,CENT,&geom) ;
					#endif
				}
			}
		}
	}

    // make sure there is no inflow at the inner boundary 
	if (N1_MPI_offset == 0){
		for (i = -N1G; i <= -1; i++){  
			#pragma omp parallel shared(prim, i) private(j,z)
			{
				#pragma omp for schedule(static,1)	
				for (j = N2_MPI_offset-2; j < N2_MPI_offset + N2_MPI+2; j++){
					for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
						inflow_check(prim[index(-1, j, z)], i, j, 0);
						inflow_check(prim[index(-2, j, z)], i, j, 0);
					}
				}
			}
		}
	}
	// make sure there is no inflow at the outer boundary
	if (N1_MPI_offset + N1_MPI == N1){
		for (i = N1; i <= N1 + N1G - 1; i++){
			#pragma omp parallel shared(prim, i) private(j,z)
			{
			#pragma omp for schedule(static,1)
				for (j = N2_MPI_offset - N2G; j < N2_MPI_offset + N2_MPI + N2G; j++){
					for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
						inflow_check(prim[index(N1, j, z)], i, j, 1);
						inflow_check(prim[index(N1 + 1, j, z)], i, j, 1);
					}
				}
			}
		}
	}

	//copy all densities and B^phi in; interpolate linearly transverse velocity
	#if(POLEFIX && POLEFIX < N2/2)
	jref = POLEFIX;
	if (N2_MPI_offset == 0){
		#pragma omp parallel shared(prim, jref) private(i,j,z,k)
		{
		#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2)
							//don't touch magnetic fields
							continue;
							else if (k == U2) {
								//linear interpolation of transverse velocity (both poles)
								prim[index(i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[index(i, jref, z)][k];
							}
							else {
								//everything else copy (both poles)
								prim[index(i, j, z)][k] = prim[index(i, jref, z)][k];
							}
						}
					}
				}
			}
		}
	}
	if (N2_MPI_offset + N2_MPI == N2){
		#pragma omp parallel shared(prim, jref) private(i,j,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2)
							//don't touch magnetic fields
							continue;
							else if (k == U2) {
								//linear interpolation of transverse velocity (both poles)
								prim[index(i, N2 - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[index(i, N2 - 1 - jref, z)][k];
							}
							else {
								//everything else copy (both poles)
								prim[index(i, N2 - 1 - j, z)][k] = prim[index(i, N2 - 1 - jref, z)][k];
							}
						}
					}
				}
			}
		}
	}
	#endif

    // polar BCs 
	if (N2_MPI_offset == 0){	
		#pragma omp parallel shared(prim, pflag) private(i,j,z, k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
					//#pragma omp simd
					PLOOP{
						prim[index(i, -1, z)][k] = prim[index(i, 0, z)][k];
						prim[index(i, -2, z)][k] = prim[index(i, 1, z)][k];
					}
					pflag[index(i, -1, z)] = pflag[index(i, 0, z)];
					//pflag[index(i, -2, z)] = pflag[index(i, 1, z)];
				}
			}
		}
	}

	if (N2_MPI_offset + N2_MPI == N2){
		#pragma omp parallel shared(prim, pflag) private(i,z, k)
		{
		#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
					//#pragma omp simd
					PLOOP{
						prim[index(i, N2, z)][k] = prim[index(i, N2-1, z)][k];
						prim[index(i, N2 + 1, z)][k] = prim[index(i, N2-2, z)][k];
					}
					pflag[index(i, N2, z)] = pflag[index(i, N2-1, z)];
					//pflag[index(i, N2 + 1, z)] = pflag[index(i, N2-2, z)];
				}
			}
		}
	}

    // make sure b and u are antisymmetric at the poles 
	if (N2_MPI_offset == 0){
		#pragma omp parallel shared(prim) private(i,j,z)
		{
			#pragma omp for schedule(static,1)
				
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (j = -N2G; j < 0; j++) {
					for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
						prim[index(i, j, z)][U2] *= -1.;
						prim[index(i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
	if (N2_MPI_offset + N2_MPI == N2){
		#pragma omp parallel shared(prim) private(i,j,z)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - N1G; i < N1_MPI_offset + N1_MPI + N1G; i++){
				for (j = N2; j < N2 + N2G; j++) {
					for (z = -N3G + N3_MPI_offset; z < N3_MPI + N3_MPI_offset + N3G; z++) {
						prim[index(i, j, z)][U2] *= -1.;
						prim[index(i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}

	#if(N3G>0)
	/* phi BCs */
	//Inner phi-boundary
	if (N3_MPI_offset == 0){
		#pragma omp parallel shared(prim, pflag) private(i,j,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = -N1G + N1_MPI_offset; i < N1_MPI+N1_MPI_offset + N1G; i++){
				for (j = -N2G + N2_MPI_offset; j < N2_MPI + N2_MPI_offset + N2G; j++){
					for (z = -N3G; z < 0; z++){
						//periodic by default
						//#pragma omp simd
						PLOOP prim[index(i, j, z)][k] = prim[index(i, j, N3 + z)][k];
						//isdis[i][j][kNg] = isdis[i][j][N3+kNg];
					}
					pflag[index(i, j, -N3G + 1)] = pflag[index(i, j, N3 + -N3G + 1)];
				}
			}
		}
	}

	//Outer phi-boundary
	if (N3_MPI_offset + N3_MPI == N3){
		#pragma omp parallel shared(prim, pflag) private(i,j,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = -N1G + N1_MPI_offset; i < N1_MPI + N1_MPI_offset + N1G; i++){
				for (j = -N2G + N2_MPI_offset; j < N2_MPI + N2_MPI_offset + N2G; j++){
					for (z = 0; z < N3G; z++){
						//periodic by default
						//#pragma omp simd
						PLOOP prim[index(i, j, N3 + z)][k] = prim[index(i, j, z)][k];
						//isdis[i][j][kNg] = isdis[i][j][N3+kNg];
					}
					pflag[index(i, j, N3)] = pflag[index(i, j, 0)];
				}
			}
		}
	}
	#endif

	if ((n_rows > 1 || n_columns > 1) && MPI==1){
		bound_send(prim);
		bound_rec(prim);
	}
}

/*Send boundaries between compute nodes through MPI*/
void bound_send(double(*prim)[NPR]){
	int i, j, z, k;
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		begin2 = clock();
	}
	//Exchange boundary cells for MPI threads
	//Positive X1
	if (n1_MPI + 1 < n_columns){
		for (i = aN1_MPI[rank] - 2; i < aN1_MPI[rank]; i++){
			for (j = -2; j < aN2_MPI[rank] + 2; j++){
				for (z = -N3G; z < aN3_MPI[rank] + N3G; z++){
					for (k = 0; k < NPR; k++){
						send2[NPR*(i - (aN1_MPI[rank] - 2))*(N3_MPI + 2 * N3G)*(aN2_MPI[rank] + 4) + NPR*(j + 2)*(N3_MPI + 2 * N3G) + NPR*z +k] =
							prim[index(i + aN1_MPI_offset[rank], j + aN2_MPI_offset[rank], z + aN3_MPI_offset[rank])][k];
					}
				}
			}
		}
		rc = MPI_Irecv(&receive4[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank + n_rows] + 4) * 2, MPI_DOUBLE, rank + n_rows, 4, MPI_COMM_WORLD, &reqs[3]);
		rc = MPI_Isend(&send2[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank] + 4) * 2, MPI_DOUBLE, rank + n_rows, 2, MPI_COMM_WORLD, &reqs[0]);
	}
	//Negative X1
	if (n1_MPI - 1 >= 0){
		for (i = 0; i < 2; i++){
			for (j = -2; j < aN2_MPI[rank] + 2; j++){
				for (z = -N3G; z < aN3_MPI[rank] + N3G; z++){
					for (k = 0; k < NPR; k++){
						send4[NPR*(i)*(N3_MPI + 2 * N3G)*(aN2_MPI[rank] + 4) + NPR*(j + 2)*(N3_MPI + 2 * N3G) + NPR*z+k] =
							prim[index(i + aN1_MPI_offset[rank], j + aN2_MPI_offset[rank], z + aN3_MPI_offset[rank])][k];
					}
				}
			}
		}
		rc = MPI_Irecv(&receive2[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank - n_rows] + 4) * 2, MPI_DOUBLE, rank - n_rows, 2, MPI_COMM_WORLD, &reqs[1]);
		rc = MPI_Isend(&send4[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank] + 4) * 2, MPI_DOUBLE, rank - n_rows, 4, MPI_COMM_WORLD, &reqs[2]);
	}
	//Positive X2
	if (n2_MPI + 1 < n_rows){
		for (j = aN2_MPI[rank] - 2; j < aN2_MPI[rank]; j++){
			for (i = -2; i < aN1_MPI[rank] + 2; i++){
				for (z = -N3G; z < aN3_MPI[rank] + N3G; z++){
					for (k = 0; k < NPR; k++){
						send3[NPR*(j - (aN2_MPI[rank] - 2))*(N3_MPI + 2 * N3G)*(aN1_MPI[rank] + 4) + NPR*(i + 2)*(N3_MPI + 2 * N3G) + NPR*z + k] =
							prim[index(i + aN1_MPI_offset[rank], j + aN2_MPI_offset[rank], z + aN3_MPI_offset[rank])][k];
					}
				}
			}
		}
		rc = MPI_Irecv(&receive1[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank + 1] + 4) * 2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[7]);
		rc = MPI_Isend(&send3[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank] + 4) * 2, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &reqs[4]);
	}
	//Negative X2
	if (n2_MPI - 1 >= 0){
		for (j = 0; j < 2; j++){
			for (i = -2; i < aN1_MPI[rank] + 2; i++){
				for (z = -N3G; z < aN3_MPI[rank] + N3G; z++){
					for (k = 0; k < NPR; k++){
						send1[NPR*(j)*(N3_MPI + 2 * N3G)*(aN1_MPI[rank] + 4) + NPR*(i + 2)*(N3_MPI + 2 * N3G) + NPR*z + k] =
							prim[index(i + aN1_MPI_offset[rank], j + aN2_MPI_offset[rank], z + aN3_MPI_offset[rank])][k];
					}
				}
			}
		}
		rc = MPI_Irecv(&receive3[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank - 1] + 4) * 2, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &reqs[5]);
		rc = MPI_Isend(&send1[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank] + 4) * 2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[6]);
	}
}

/*Receive boundaries for compute nodes through MPI*/
void bound_rec(double(*prim)[NPR]){
	int i, j, z, k;

	//positive X1
	if (n1_MPI - 1 >= 0){
		//rc = MPI_Irecv(&receive2[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank - n_rows] + 4) * 2, MPI_DOUBLE, rank - n_rows, 2, MPI_COMM_WORLD, &reqs[1]);
		MPI_Wait(&reqs[1], Stat);
		for (i = aN1_MPI[rank - n_rows] - 2; i < aN1_MPI[rank - n_rows]; i++){
			for (j = -2; j < aN2_MPI[rank - n_rows] + 2; j++){
				for (z = -N3G; z < aN3_MPI[rank - n_rows] + N3G; z++){
					for (k = 0; k < NPR; k++){
						prim[index(i + aN1_MPI_offset[rank - n_rows], j + aN2_MPI_offset[rank - n_rows], z + aN3_MPI_offset[rank - n_rows])][k] = 
							receive2[NPR*(i - (aN1_MPI[rank - n_rows] - 2))*(N3_MPI + 2 * N3G)*(aN2_MPI[rank - n_rows] + 4) + NPR*(j + 2)*(N3_MPI + 2 * N3G) + NPR*z+k];
					}
				}
			}
		}
	}
	//Negative X1
	if (n1_MPI + 1 < n_columns){
		//rc = MPI_Irecv(&receive4[0], NPR*(N3_MPI + 2 * N3G)*(aN2_MPI[rank + n_rows] + 4) * 2, MPI_DOUBLE, rank + n_rows, 4, MPI_COMM_WORLD, &reqs[3]);
		MPI_Wait(&reqs[3], Stat);
		for (i = 0; i < 2; i++){
			for (j = -2; j < aN2_MPI[rank + n_rows] + 2; j++){
				for (z = -N3G; z < aN3_MPI[rank + n_rows] + N3G; z++){
					for (k = 0; k < NPR; k++){
						prim[index(i + aN1_MPI_offset[rank + n_rows], j + aN2_MPI_offset[rank + n_rows], z + aN3_MPI_offset[rank + n_rows])][k] = 
							receive4[NPR*(i)*(N3_MPI + 2 * N3G)*(aN2_MPI[rank + n_rows] + 4) + NPR*(j + 2)*(N3_MPI + 2 * N3G) + NPR*z+k];
					}
				}
			}
		}
	}
	//Positive X2
	if (n2_MPI - 1 >= 0){
		//rc = MPI_Irecv(&receive3[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank - 1] + 4) * 2, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &reqs[5]);
		MPI_Wait(&reqs[5], Stat);
		for (j = aN2_MPI[rank - 1] - 2; j < aN2_MPI[rank - 1]; j++){
			for (i = -2; i < aN1_MPI[rank - 1] + 2; i++){
				for (z = -N3G; z < aN3_MPI[rank - 1] + N3G; z++){
					for (k = 0; k < NPR; k++){
						prim[index(i + aN1_MPI_offset[rank - 1], j + aN2_MPI_offset[rank - 1], z + aN3_MPI_offset[rank - 1])][k] = 
							receive3[NPR*(j - (aN2_MPI[rank - 1] - 2))*(N3_MPI + 2 * N3G)*(aN1_MPI[rank - 1] + 4) + NPR*(i + 2)*(N3_MPI + 2 * N3G) + NPR*z+k];
					}
				}
			}
		}
	}
	//Negative X2
	if (n2_MPI + 1 < n_rows){
		//rc = MPI_Irecv(&receive1[0], NPR*(N3_MPI + 2 * N3G)*(aN1_MPI[rank + 1] + 4) * 2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[7]);
		MPI_Wait(&reqs[7], Stat);
		for (j = 0; j < 2; j++){
			for (i = -2; i < aN1_MPI[rank + 1] + 2; i++){
				for (z = -N3G; z < aN3_MPI[rank + 1] + N3G; z++){
					for (k = 0; k < NPR; k++){
						prim[index(i + aN1_MPI_offset[rank + 1], j + aN2_MPI_offset[rank + 1], z + aN3_MPI_offset[rank + 1])][k] = 
							receive1[NPR*(j)*(N3_MPI + 2 * N3G)*(aN1_MPI[rank + 1] + 4) + NPR*(i + 2)*(N3_MPI + 2 * N3G) + NPR*z+k];
					}
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		end2 = clock();
		time_spent3 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	}
}


void inflow_check(double *pr, int ii, int jj, int type ){
    struct of_geom geom ;
    double ucon[NDIM] ;
    int j,k ;
    double alpha,beta1,gamma,vsq ;

    get_geometry(ii,jj,CENT,&geom) ;
    ucon_calc(pr, &geom, ucon) ;

    if( ((ucon[1] > 0.) && (type==0)) || ((ucon[1] < 0.) && (type==1)) ) { 
		/* find gamma and remove it from primitives */
		if( gamma_calc(pr,&geom,&gamma) ) { 
			fflush(stderr);
			fprintf(stderr,"\ninflow_check(): gamma failure \n");
			fflush(stderr);
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

