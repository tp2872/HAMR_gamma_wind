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

/* all diagnostics subroutine */
void diag(int call_code);
void diag(int call_code)
{
	char dfnam[100],ifnam[100] ;
	int i,j,z,k,l ;
	FILE *dump_file;
	double U[NPR],pp,e,rmed,divb,divbmax,e_fin,m_fin,gamma ;
	struct of_geom geom ;
	struct of_state q ;
	int imax,jmax,zmax;
	static double e_init,m_init ;
	static FILE *ener_file ;
	int di, dj, dz;
	
	di = (N1>1);
	dj = (N2>1);
	dz = (N3>1);

	if (rank == 0){
		if (call_code == INIT_OUT) {
			/* set things up */
			ener_file = fopen("ener.out", "a");
			if (ener_file == NULL) {
				fprintf(stderr, "error opening energy output file\n");
				exit(1);
			}
		}
		
	}

	/* calculate conserved quantities */
	if(call_code==INIT_OUT || 
	   call_code==LOG_OUT ||
	   call_code==FINAL_OUT &&
	   !failed) {
		pp = 0. ;
		e = 0. ;
		rmed = 0. ;
		divbmax = 0. ;
		imax = 0 ; 
		jmax = 0 ;
	
		ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) {
			get_geometry(i,j,CENT,&geom) ;
			get_state(p[index(i,j,z)],&geom,&q) ;
			primtoU(p[index(i,j,z)],&q,&geom,U) ;

			rmed += U[RHO]*dV ;
			pp += U[U3]*dV ;
			e += U[UU]*dV ;

			/* flux-ct defn */
			divb = fabs(
				#if(N1>1)
				0.25*(
				+p[index(i,j,z)][B1] * gdet[index2(i,j)][CENT]
				+ p[index(i, j, z - dz)][B1] * gdet[index2(i, j)][CENT]
				+ p[index(i, j - dj, z)][B1] * gdet[index2(i, j-dj)][CENT]
				+ p[index(i, j - dj, z - dz)][B1] * gdet[index2(i, j-dj)][CENT]
				- p[index(i - 1, j, z)][B1] * gdet[index2(i-1, j)][CENT]
				- p[index(i - 1, j, z - dz)][B1] * gdet[index2(i-1, j)][CENT]
				- p[index(i - 1, j - dj, z)][B1] * gdet[index2(i-1, j-dj)][CENT]
				- p[index(i - 1, j - dj, z - dz)][B1] * gdet[index2(i-1, j-dj)][CENT]
				) / dx[1]
				#endif
				#if(N2>1)
				+ 0.25*(
				+p[index(i, j, z)][B2] * gdet[index2(i, j)][CENT]
				+ p[index(i, j, z - dz)][B2] * gdet[index2(i, j)][CENT]
				+ p[index(i - di, j, z)][B2] * gdet[index2(i-di, j)][CENT]
				+ p[index(i - di, j, z - dz)][B2] * gdet[index2(i-di, j)][CENT]
				- p[index(i, j - 1, z)][B2] * gdet[index2(i, j-1)][CENT]
				- p[index(i, j - 1, z - dz)][B2] * gdet[index2(i, j-1)][CENT]
				- p[index(i - di, j - 1, z)][B2] * gdet[index2(i-di, j-1)][CENT]
				- p[index(i - di, j - 1, z - dz)][B2] * gdet[index2(i-di, j-1)][CENT]
				) / dx[2]
				#endif
				#if(N3>1)
				+ 0.25*(
				+p[index(i, j, z)][B3] * gdet[index2(i, j)][CENT]
				+ p[index(i - di, j, z)][B3] * gdet[index2(i-di, j)][CENT]
				+ p[index(i, j - dj, z)][B3] * gdet[index2(i, j-dj)][CENT]
				+ p[index(i - di, j - dj, z)][B3] * gdet[index2(i-di, j-dj)][CENT]
				- p[index(i, j, z - 1)][B3] * gdet[index2(i, j)][CENT]
				- p[index(i - di, j, z - 1)][B3] * gdet[index2(i-di, j)][CENT]
				- p[index(i, j - dj, z - 1)][B3] * gdet[index2(i, j-dj)][CENT]
				- p[index(i - di, j - dj, z - 1)][B3] * gdet[index2(i-di, j-dj)][CENT]
				) / dx[3]
				#endif
				);
			if (divb > divbmax && i > 0 && j > N2_MPI / 10  && (z>0 || N3 == 1)) {
				imax = i ;
				jmax = j ;
				zmax = z;
				divbmax = divb ;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank != 0){
			rc = MPI_Isend(&e, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank]);
			rc = MPI_Isend(&rmed, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &reqs[rank+numtasks]);
			rc = MPI_Isend(&pp, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &reqs[rank + 2*numtasks]);
			rc = MPI_Isend(&divbmax, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &reqs[rank + 3*numtasks]);
		}
		else{
			double inmsg;
			for (i = 1; i < numtasks; i++){
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, Stat);
				e += inmsg;
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, Stat);
				rmed += inmsg;
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, Stat);
				pp += inmsg;
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, Stat);
				if (inmsg > divbmax){
					divbmax = inmsg;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
	}

	if (rank == 0){
		if (call_code == INIT_OUT) {
			e_init = e;
			m_init = rmed;
		}

		if (call_code == FINAL_OUT) {
			e_fin = e;
			m_fin = rmed;
			fprintf(stderr, "\n\nEnergy: ini,fin,del: %g %g %g\n",
				e_init, e_fin, (e_fin - e_init) / e_init);
			fprintf(stderr, "mass: ini,fin,del: %g %g %g\n",
				m_init, m_fin, (m_fin - m_init) / m_init);
		}

		if (call_code == INIT_OUT ||
			call_code == LOG_OUT ||
			call_code == FINAL_OUT) {
			fprintf(stderr, "LOG      t=%g \t divbmax: %d %d %d %g\n",
				t, imax, jmax, zmax, divbmax);
			//fprintf(ener_file, "%10.5g %10.5g %10.5g %10.5g %15.8g %15.8g ",
			//	t, rmed, pp, e, p[index(N1 / 2, N2 / 2, 0)][UU] * pow(p[index(N1 / 2, N2 / 2, 0)][RHO], -gam),
			//	p[index(N1 / 2, N2 / 2, 0)][UU]);
			fprintf(ener_file, "%15.8g %15.8g %15.8g ", mdot, edot, ldot);
			fprintf(ener_file, "\n");
			fflush(ener_file);
		}
	}


		/* gdump only at code start */
		if (call_code == INIT_OUT) {
			// make grid dump file 
			sprintf(dfnam, "dumps/gdump");
			if (rank == 0){
				fprintf(stderr, "GDUMP    file=%s\n", dfnam);
			}
			
			for (i = 0; i < numtasks; i++){
				if (rank == i){
					if (rank == 0){
						dump_file = fopen(dfnam, "wb");
					}
					else{
						dump_file = fopen(dfnam, "ab");
					}
					if (dump_file == NULL) {
						fprintf(stderr, "error opening grid dump file\n");
						exit(2);
					}
					gdump(dump_file);
					fclose(dump_file);
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
			#if(!GPU_BENCHMARK && !GPU_DEBUG && ! CPU_OPENMP)
			free(dq);
			#endif
		}

		// dump at regular intervals 
		if (call_code == INIT_OUT ||
			call_code == DUMP_OUT ||
			call_code == FINAL_OUT) {
			// make regular dump file 
			sprintf(dfnam, "dumps/dump%03d", dump_cnt);
			if (rank == 0){
				fprintf(stderr, "DUMP     file=%s\n", dfnam);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			for (i = 0; i < numtasks; i++){
				if (rank == i){
					if (rank == 0){
						dump_file = fopen(dfnam, "wb");
					}
					else{
						dump_file = fopen(dfnam, "ab");
					}
					if (dump_file == NULL) {
						fprintf(stderr, "error opening dump file\n");
						exit(2);
					}
					dump(dump_file);
					fclose(dump_file);
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}

			dump_cnt++;
		}

		/* image dump at regular intervals */
		
			if (call_code == IMAGE_OUT ||
				call_code == INIT_OUT ||
				call_code == FINAL_OUT) {
				int rank_cnt;
				
				/*Copy primitive variables to main node 0 for image output*/
				MPI_Barrier(MPI_COMM_WORLD);
				if (rank > 0){
					ZSLOOP(-N1G , N1_MPI + 1, -N2G, N2_MPI +1){
						for (k = 0; k < NPR; k++){
							send[k*(N2_MPI + 4)*(N1_MPI + 4) + (i + 2)*(N2_MPI + 4) + (j + 2)] = p[index(i + N1_MPI_offset, j + N2_MPI_offset, 0)][k];
						}
					}
					rc = MPI_Send(&send[0], NPR*(N1_MPI + 4)*(N2_MPI + 4), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
				}
				else{
					ZSLOOP(-N1G + N1_MPI_offset, N1_MPI + N1_MPI_offset - 1 + N1G, -N2G + N2_MPI_offset, N2_MPI + N2_MPI_offset - 1 + N2G){
						for (k = 0; k < NPR; k++){
							psave[index3(i, j)][k] = p[index(i, j, 0)][k];
						}
					}
					for (rank_cnt = 1; rank_cnt < numtasks; rank_cnt++){
						rc = MPI_Recv(&receive[0], NPR*(aN1_MPI[rank_cnt] + 4)*(aN2_MPI[rank_cnt] + 4), MPI_DOUBLE, rank_cnt, rank_cnt, MPI_COMM_WORLD, Stat);
						ZSLOOP(-N1G, aN1_MPI[rank_cnt] - 1 + N1G, -N2G, aN2_MPI[rank_cnt] - 1 + N2G){
							for (k = 0; k < NPR; k++){
								psave[index3(i + aN1_MPI_offset[rank_cnt], j + aN2_MPI_offset[rank_cnt])][k] = receive[k*(aN2_MPI[rank_cnt] + 4)*(aN1_MPI[rank_cnt] + 4) + (i + 2)*(aN2_MPI[rank_cnt] + 4) + (j + 2)];
							}
						}
					}
				}

				MPI_Barrier(MPI_COMM_WORLD);
				/*Copy failure points to main node 0 for image output*/
				if (rank > 0){

					ZSLOOP(-2, N1_MPI + 1, -2, N2_MPI + 1){
						for (k = 0; k < NFAIL; k++){
							send[k*(N2_MPI + 4)*(N1_MPI + 4) + (i + 2)*(N2_MPI + 4) + (j + 2)] = 0.0;
						}
						for (z = 0; z <= N3_MPI - 1; z++){
							#pragma simd
							for (k = 0; k < NFAIL; k++){
								send[k*(N2_MPI + 4)*(N1_MPI + 4) + (i + 2)*(N2_MPI + 4) + (j + 2)] += (double)failimage[index(i + N1_MPI_offset, j + N2_MPI_offset, z + N3_MPI_offset)][k];
							}
						}
					}
					rc = MPI_Send(&send[0], NFAIL*(N1_MPI + 4)*(N2_MPI + 4), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
				}
				else{
					ZSLOOP(-N1G + N1_MPI_offset, N1_MPI + N1_MPI_offset -1 + N1G, -N2G + N2_MPI_offset, N2_MPI + N2_MPI_offset - 1 +N2G){
						#pragma simd
						for (k = 0; k < NFAIL; k++){
							fsave[index3(i, j)][k] =0.0;
						}
						for (z = -N3G; z <= N3_MPI - 1 + N3G; z++){
							#pragma simd
							for (k = 0; k < NFAIL; k++){
								fsave[index3(i, j)][k] += (double)failimage[index(i, j, z)][k];
							}
						}
					}
					for (rank_cnt = 1; rank_cnt < numtasks; rank_cnt++){
						rc = MPI_Recv(&receive[0], NFAIL*(aN1_MPI[rank_cnt] + 4)*(aN2_MPI[rank_cnt] + 4), MPI_DOUBLE, rank_cnt, rank_cnt, MPI_COMM_WORLD, Stat);
						ZSLOOP(-N1G, aN1_MPI[rank_cnt] - 1 + N1G, -N2G, aN2_MPI[rank_cnt] - 1 +N2G){
							for (k = 0; k < NFAIL; k++){
								fsave[index3(i + aN1_MPI_offset[rank_cnt], j + aN2_MPI_offset[rank_cnt])][k] = receive[k*(aN2_MPI[rank_cnt] + 4)*(aN1_MPI[rank_cnt] + 4) + (i + 2)*(aN2_MPI[rank_cnt] + 4) + (j + 2)];
							}
						}
					}
				}
				
				if (rank == 0){
					image_all(image_cnt);
				}
				MPI_Barrier(MPI_COMM_WORLD);
				image_cnt++;
			}
}


/** some diagnostic routines **/


void fail(int fail_type)
{

	failed = 1 ;

	fprintf(stderr,"\n\nfail: %d %d %d\n",icurr,jcurr,fail_type) ;

	area_map(icurr,jcurr,p) ;
	
	fprintf(stderr,"fail\n") ;

	diag(FINAL_OUT) ;

	/* for diagnostic purposes */
	exit(0) ;
}



/* map out region around failure point */
void area_map(int i, int j, double (*prim)[NPR])
{
	int k ;

	fprintf(stderr,"area map\n") ;

	PLOOP {
		fprintf(stderr,"variable %d \n",k) ;
		fprintf(stderr,"i = \t %12d %12d %12d\n",i-1,i,i+1) ;
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j+1,
				prim[index(i-1, j+1, 0)][k],
				prim[index(i, j + 1, 0)][k],
				prim[index(i + 1, j + 1, 0)][k]);
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j,
				prim[index(i - 1, j, 0)][k],
				prim[index(i, j , 0)][k],
				prim[index(i + 1, j, 0)][k]);
		fprintf(stderr,"j = %d \t %12.5g %12.5g %12.5g\n",
				j-1,
				prim[index(i - 1, j - 1, 0)][k],
				prim[index(i, j - 1, 0)][k],
				prim[index(i +1, j - 1, 0)][k]);
	}

	/* print out other diagnostics here */

}

/* evaluate fluxed based diagnostics; put results in
 * global variables */

void diag_flux(double (*F1)[NPR])
{
	int i, j, z;
    mdot = edot = ldot = 0. ;
	if (n1_MPI == 0){
		#pragma omp parallel shared(F1, dx, mdot, edot, ldot) private(j)
		{
			#pragma omp for schedule(dynamic,1)
			for (j = N2_MPI_offset; j < N2_MPI_offset + N2_MPI; j++) {
				for (z = N3_MPI_offset; z < N3_MPI_offset + N3_MPI; z++) {
					#pragma omp critical
					{
						mdot += F1[index(0,j,z)][RHO] * 2.*M_PI*dx[2];
					}
					#pragma omp critical
					{
						edot -= (F1[index(0, j, z)][UU] - F1[index(0, j, z)][RHO])*2.*M_PI*dx[2];
					}
					#pragma omp critical
					{
						ldot += F1[index(0, j, z)][U3] * 2.*M_PI*dx[2];
					}
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank != 0){
			rc = MPI_Isend(&mdot, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &reqs[rank]);
			rc = MPI_Isend(&edot, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &reqs[rank+numtasks]);
			rc = MPI_Isend(&ldot, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &reqs[rank+2*numtasks]);
		}
		else{
			double inmsg;
			for (i = 1; i < n_rows; i++){
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, Stat);
				mdot += inmsg;
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, Stat);
				edot += inmsg;
				rc = MPI_Recv(&inmsg, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, Stat);
				ldot += inmsg;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

