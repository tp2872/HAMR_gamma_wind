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
        Astrophysical Journal1, 626.

   
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
#include "defs.h"
#include "cudaProfiler.h"

/*****************************************************************/
/*****************************************************************
   main():
   ------

     -- Initializes, time-steps, and concludes the simulation. 
     -- Handles timing of output routines;
     -- Main is main, what more can you say.  

-*****************************************************************/
int main(int argc, char *argv[])
{
	double tdump, timage, tlog;
	int nfailed = 0;
	int i, j, u, n;
	double r, th, phi, X[NDIM];
	int threadid;

	/* Perform Initializations, either directly or via checkpoint */
	MPI_initialize(argc, argv);
	#pragma omp parallel shared(nthreads) private(threadid)
	{
		threadid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		if (threadid == 0 && rank == 0) {
			fprintf(stderr, "nthreads = %d\n", nthreads);
		}
	}
	//omp_set_num_threads(1);

	#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	GPU_init();
	#endif

	set_AMR();
	if (rank == 0){
		system("mkdir dumps gdumps rdumps0 rdumps1");
	}

	nstep = 0;
	defcon = 1.;

	if (!restart_read()) {
		init();
		int refined = 0;
		int derefined = 0;
		#if(DEREFINE_POLE)
		derefine_pole();
		#endif
	}

	/* do initial diagnostics */
	#if(NONSYMMETRIC)
	for (n = 0; n < n_active; n++) set_grid(n_ord[n]);
	#endif

	activate_blocks();

	balance_load();
	#if(GPU_ENABLED)
	balance_load_gpu();
	#endif

	/* do initial diagnostics */
	#if (DIAG_ON)
	first_dump = 0;
	diag(INIT_OUT);
	#endif

	DTl = 20.0;
	bound_prim(p, 1);
	tdump = t + DTd;
	timage = t + DTi;
	tlog = t + DTl;
	tref = t + TREF;
	defcon = 1. ;
	time_spent3 = 0.0;
	begin1 = time(NULL);
	//cuProfilerStart();

	while(t < tf) {
		/* step variables forward in time */
		nstroke = 0 ;	

		/*Used for performance analysis*/
		#if(GPU_BENCHMARK)
		GPU_benchmark();
		#endif

		/*Used for running OpenCL on either GPU or CPU*/
		#if(GPU_ENABLED && !GPU_DEBUG)
		GPU_step_ch();
		#endif
		#if(CPU_OPENMP)
		step_ch();
		#endif

		/*Used for debugging*/
		#if(GPU_DEBUG)
		step_ch_debug();
		#endif

		if (t >= tref && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			#if(!DEREFINE_POLE)
			#if (OpenCL_enable==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			bound_prim(p, 1);
			check_refcrit();
			if (rank == 0) printf("Refinement succesfull! \n");
			#endif
			tref += TREF;
		}

		/* Handle output frequencies: */
		if (t >= tdump && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			#if (OpenCL_enable==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			diag(DUMP_OUT) ;
			tdump += DTd;
		}

		if (t >= tlog && nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) {
			#if (OpenCL_enable==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			
			if (dt>2.) break;
			restart_write(); //do restart dumb simultaneous with log
			tlog +=  DTl;
		}			

		#if TIMER
		if (nstep % (2*320) == 0){
			#if (OpenCL_enable == 1)
			#endif
			end1 = time(NULL);
			#if (OpenCL_enable==1)
			for (n = 0; n < n_active; n++) GPU_read(n_ord[n]);
			#endif
			bound_prim(p, 1);
			diag(LOG_OUT);
			MPI_Allreduce(MPI_IN_PLACE, &ndt1, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt2, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			MPI_Allreduce(MPI_IN_PLACE, &ndt3, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
			if (rank == 0){
				fprintf(stderr, "Runtime: %f ", (double)(end1 - begin1));
				fprintf(stderr, "MPI-time: %f ", time_spent3);
				fprintf(stderr, "dt1: %f ", ndt1);
				fprintf(stderr, "dt2: %f ", ndt2);
				fprintf(stderr, "dt3: %f \n", ndt3);
				fprintf(stderr, "nstep: %d \n", nstep);
				fflush(stderr);
			}
			time_spent3 = 0.0;
			begin1 = time(NULL);
		}
		#endif
		//cuProfilerStop();
		/* deal with failed timestep, though we usually exit upon failure */
		if(failed) {
			restart_read() ;
			failed = 0 ;
			nfailed = nstep ;
			defcon = 0.3 ;
		}
		if(nstep > nfailed + DTr*4.*(1 + 1./defcon)) defcon = 1. ;
	}
	fprintf(stderr,"ns,ts: %d %d\n",nstep,nstep*N1*N2) ;

	/* do final diagnostics */
	#if (DIAG_ON)
	diag(FINAL_OUT) ;
	#endif

	/*Close GPU*/
	for (n=0; n<n_active; n++) GPU_finish(n_ord[n]);
	return(0) ;
}

//Set row major order in case of derfinement near pole
int rm_order(void){
	int l, i, j, z, ni, nj, nz;
	int number = 0;
	int number_node = 0;
	ni = NB_1*pow(1 + REF_1, N_LEVELS - 1);
	nj = NB_2*pow(1 + REF_2, N_LEVELS - 1);
	nz = NB_3*pow(1 + REF_3, N_LEVELS - 1);
	for (z = 0; z < nz; z++)for (j = 0; j < nj; j++)for (i = 0; i < ni; i++){
		for (l = 0; l<N_LEVELS; l++){
			if (i<NB_1*pow(1 + REF_1, l) && j<NB_2*pow(1 + REF_2, l) && z<NB_3*pow(1 + REF_3, l) && block[AMR_coord_linear(l, i, j, z)][AMR_ACTIVE] == 1){
				n_ord_total_RM[number] = AMR_coord_linear(l, i, j, z);
				number++;
				if (block[AMR_coord_linear(l, i, j, z)][AMR_NODE] == rank){
					n_ord_RM[number_node] = AMR_coord_linear(l, i, j, z);
					number_node++;
				}
				block[AMR_coord_linear(l, i, j, z)][RM_ORDER] = number;
			}
		}
	}
}

//This function derefines in z near the pole
int derefine_pole(void){
	int i, j, z, l, ni, nj, nz;
	if (REF_3 != 1 || REF_1 == 1 || REF_2 == 1){
		fprintf(stderr,"Error! Derefinement near the pole works only for REF_1=0, REF_2=0, REF_3=1 \n");
		return -1;
	}
	if (NB_2 % 6 != 0){
		fprintf(stderr,"For derefinement near the pole chose NB_2 6, 12, 24,48 for 1, 2, 3, 4 levels of derefinement near the pole! \n");
		return -1;
	}
	if (calc_mem(NB_1*NB_2*NB_3*pow(2.,N_LEVELS-1))>((double)numtasks*(double)(N_GPU)* 4. * (pow(10., 9.))) && rank == 1) fprintf(stderr, "You are exceeding the maximum memory size of 4 GB per GPU by refining too many blocks! Code will probably segfault, choose a bigger cluster \n");

	for (l = 0; l<N_LEVELS-1; l++){
		pre_refine();
		ni = NB_1*pow(1 + REF_1, l);
		nj = NB_2*pow(1 + REF_2, l);
		nz = NB_3*pow(1 + REF_3, l);
		for (i = 0; i < ni; i++)for (j = pow(2, l); j < nj - (pow(2, l)); j++)for (z = 0; z < nz; z++){
			if ((double)pow(2, l) < 0.25*NB_2){
				refine(AMR_coord_linear(l, i, j, z));
			}
		}
		MPI_Barrier(mpi_cartcomm);
		post_refine();
		if (rank == 0)fprintf(stderr, "Derefinement at level %d complete! \n", l);
	}

	//Set boundary conditions
	if (rank == 0) fprintf(stderr, "Bounding AMR blocks, watch out for errors or divb increasing! \n");
	//bound_prim(p, 1);
	//#if(GPU_ENABLED || GPU_DEBUG || GPU_BENCHMARK)
	//GPU_boundprim(1);
	//#endif
	return 1;
}

/*****************************************************************/
/*****************************************************************
  set_arrays():
  ----------

       -- sets to zero all arrays, plus performs pointer trick 
          so that grid arrays can legitimately refer to ghost 
          zone quantities at positions  i = -2, -1, N1, N1+1 and 
          j = -2, -1, N2, N2+1

 *****************************************************************/
void set_arrays_image(void)
{
	int i;
	#if(IMAGE_ON)
	//psave = (double(*)[NPR])calloc((N1 + 2 * N1G)*(N2 + 2 * N2G),sizeof(double[NPR]));
	//fsave = (double(*)[NFAIL])calloc((N1 + 2 * N1G)*(N2 + 2 * N2G),sizeof(double[NFAIL]));
	//fimage = (double(**))calloc(NIMG ,sizeof(double *));
	for (i = 0; i < NIMG; i++){
		//fimage[i] = (double(*))calloc(NIMG*N1*N2 ,sizeof(double));
	}
	#endif
}

void free_arrays(int n)
{
	int i, j, z, k;

	free(pbound[n]);
	free(p[n]);
	free(V[n]);
	#if(STAGGERED)
	free(ps[n]);
	free(psh[n]); 
	free(dE[n]);
	#endif
	free(E_corn[n]);
	free(dq[n]);
	//#if(GPU_BENCHMARK || GPU_DEBUG || CPU_OPENMP)
	free(F1[n]);
	free(F2[n]);
	free(F3[n]);
	free(ph[n]);
	//#endif
	#if(STAGGERED)
	free(stor1[n]);
	free(stor2[n]);
	#endif
	free(pflag[n]);
	free(failimage[n]);
	free(conn[n]); 
	free(gcov[n]); 
	free(gcon[n]); 
	free(gdet[n]); 
	#if(ZIRI_DUMP)
	free(dump_buffer[n]);
	free(dxdxp_z[n]); 
	free(dxpdx_z[n]); 
	#endif
	#if (ELLIPTICAL2)
	free(dU_s[n]); 
	#endif
	free(E_avg[n][0]);
	free(E_avg[n][1]);
	free(E_avg_new[n][0]);
	free(E_avg_new[n][1]);
	free(E_avg_x[n][0]);
	free(E_avg_x[n][1]);
	free(E_avg_new_x[n][0]);
	free(E_avg_new_x[n][1]);
	free(E_avg_y[n][0]);
	free(E_avg_y[n][1]);
	free(E_avg_new_y[n][0]);
	free(E_avg_new_y[n][1]);
	free(send[n]);
	free(receive[n]);
	//#if(CPU_OPENMP || PINNED==0)
	free(receive1[n]);
	free(receive2[n]);
	free(receive3[n]);
	free(receive4[n]);
	#if(N3G>0)
	free(receive5[n]);
	free(receive6[n]);
	#endif
	free(tempreceive1[n]);
	free(tempreceive2[n]);
	free(tempreceive3[n]);
	free(tempreceive4[n]);
	#if(N3G>0)
	free(tempreceive5[n]);
	free(tempreceive6[n]);
	#endif
	free(send1[n]);
	free(send2[n]);
	free(send3[n]);
	free(send4[n]);
	#if(N3G>0)
	free(send5[n]);
	free(send6[n]);
	#endif
	free(send1_3[n]);
	free(send1_4[n]);
	free(send1_7[n]);
	free(send1_8[n]);
	free(send2_1[n]);
	free(send2_2[n]);
	free(send2_3[n]);
	free(send2_4[n]);
	free(send3_1[n]);
	free(send3_2[n]);
	free(send3_5[n]);
	free(send3_6[n]);
	free(send4_5[n]);
	free(send4_6[n]);
	free(send4_7[n]);
	free(send4_8[n]);
	#if(N3G>0)
	free(send5_1[n]);
	free(send5_3[n]);
	free(send5_5[n]);
	free(send5_7[n]);
	free(send6_2[n]);
	free(send6_4[n]);
	free(send6_6[n]);
	free(send6_8[n]);
	#endif
	free(receive1_3[n]);
	free(receive1_4[n]);
	free(receive1_7[n]);
	free(receive1_8[n]);
	free(receive2_1[n]);
	free(receive2_2[n]);
	free(receive2_3[n]);
	free(receive2_4[n]);
	free(receive3_1[n]);
	free(receive3_2[n]);
	free(receive3_5[n]);
	free(receive3_6[n]);
	free(receive4_5[n]);
	free(receive4_6[n]);
	free(receive4_7[n]);
	free(receive4_8[n]);
	#if(N3G>0)
	free(receive5_1[n]);
	free(receive5_3[n]);
	free(receive5_5[n]);
	free(receive5_7[n]);
	free(receive6_2[n]);
	free(receive6_4[n]);
	free(receive6_6[n]);
	free(receive6_8[n]);
	#endif
	free(tempreceive1_3[n]);
	free(tempreceive1_4[n]);
	free(tempreceive1_7[n]);
	free(tempreceive1_8[n]);
	free(tempreceive2_1[n]);
	free(tempreceive2_2[n]);
	free(tempreceive2_3[n]);
	free(tempreceive2_4[n]);
	free(tempreceive3_1[n]);
	free(tempreceive3_2[n]);
	free(tempreceive3_5[n]);
	free(tempreceive3_6[n]);
	free(tempreceive4_5[n]);
	free(tempreceive4_6[n]);
	free(tempreceive4_7[n]);
	free(tempreceive4_8[n]);
	#if(N3G>0)
	free(tempreceive5_1[n]);
	free(tempreceive5_3[n]);
	free(tempreceive5_5[n]);
	free(tempreceive5_7[n]);
	free(tempreceive6_2[n]);
	free(tempreceive6_4[n]);
	free(tempreceive6_6[n]);
	free(tempreceive6_8[n]);
	#endif
	free(receive1_fine[n]);
	free(receive2_fine[n]);
	free(receive3_fine[n]);
	free(receive4_fine[n]);
	#if(N3G>0)
	free(receive5_fine[n]);
	free(receive6_fine[n]);
	#endif
	free(send1_fine[n]);
	free(send2_fine[n]);
	free(send3_fine[n]);
	free(send4_fine[n]);
	#if(N3G>0)
	free(send5_fine[n]);
	free(send6_fine[n]);
	#endif
	free(receive1_3fine[n]);
	free(receive1_4fine[n]);
	free(receive1_7fine[n]);
	free(receive1_8fine[n]);
	free(receive2_1fine[n]);
	free(receive2_2fine[n]);
	free(receive2_3fine[n]);
	free(receive2_4fine[n]);
	free(receive3_1fine[n]);
	free(receive3_2fine[n]);
	free(receive3_5fine[n]);
	free(receive3_6fine[n]);
	free(receive4_5fine[n]);
	free(receive4_6fine[n]);
	free(receive4_7fine[n]);
	free(receive4_8fine[n]);
	#if(N3G>0)
	free(receive5_1fine[n]);
	free(receive5_3fine[n]);
	free(receive5_5fine[n]);
	free(receive5_7fine[n]);
	free(receive6_2fine[n]);
	free(receive6_4fine[n]);
	free(receive6_6fine[n]);
	free(receive6_8fine[n]);
	#endif
	#if(CPU_OPENMP)
	free(receive1_flux[n]);
	free(receive2_flux[n]);
	free(receive3_flux[n]);
	free(receive4_flux[n]);
	free(receive5_flux[n]);
	free(receive6_flux[n]);
	free(receive1_3flux[n]);
	free(receive1_4flux[n]);
	free(receive1_7flux[n]);
	free(receive1_8flux[n]);
	free(receive2_1flux[n]);
	free(receive2_2flux[n]);
	free(receive2_3flux[n]);
	free(receive2_4flux[n]);
	free(receive3_1flux[n]);
	free(receive3_2flux[n]);
	free(receive3_5flux[n]);
	free(receive3_6flux[n]);
	free(receive4_5flux[n]);
	free(receive4_6flux[n]);
	free(receive4_7flux[n]);
	free(receive4_8flux[n]);
	#if(N3G>0)
	free(receive5_1flux[n]);
	free(receive5_3flux[n]);
	free(receive5_5flux[n]);
	free(receive5_7flux[n]);
	free(receive6_2flux[n]);
	free(receive6_4flux[n]);
	free(receive6_6flux[n]);
	free(receive6_8flux[n]);
	#endif
	free(receive1_flux1[n]);
	free(receive2_flux1[n]);
	free(receive3_flux1[n]);
	free(receive4_flux1[n]);
	free(receive5_flux1[n]);
	free(receive6_flux1[n]);
	free(receive1_3flux1[n]);
	free(receive1_4flux1[n]);
	free(receive1_7flux1[n]);
	free(receive1_8flux1[n]);
	free(receive2_1flux1[n]);
	free(receive2_2flux1[n]);
	free(receive2_3flux1[n]);
	free(receive2_4flux1[n]);
	free(receive3_1flux1[n]);
	free(receive3_2flux1[n]);
	free(receive3_5flux1[n]);
	free(receive3_6flux1[n]);
	free(receive4_5flux1[n]);
	free(receive4_6flux1[n]);
	free(receive4_7flux1[n]);
	free(receive4_8flux1[n]);
	#if(N3G>0)
	free(receive5_1flux1[n]);
	free(receive5_3flux1[n]);
	free(receive5_5flux1[n]);
	free(receive5_7flux1[n]);
	free(receive6_2flux1[n]);
	free(receive6_4flux1[n]);
	free(receive6_6flux1[n]);
	free(receive6_8flux1[n]);
	#endif
	free(receive1_3flux2[n]);
	free(receive1_4flux2[n]);
	free(receive1_7flux2[n]);
	free(receive1_8flux2[n]);
	free(receive2_1flux2[n]);
	free(receive2_2flux2[n]);
	free(receive2_3flux2[n]);
	free(receive2_4flux2[n]);
	free(receive3_1flux2[n]);
	free(receive3_2flux2[n]);
	free(receive3_5flux2[n]);
	free(receive3_6flux2[n]);
	free(receive4_5flux2[n]);
	free(receive4_6flux2[n]);
	free(receive4_7flux2[n]);
	free(receive4_8flux2[n]);
	#if(N3G>0)
	free(receive5_1flux2[n]);
	free(receive5_3flux2[n]);
	free(receive5_5flux2[n]);
	free(receive5_7flux2[n]);
	free(receive6_2flux2[n]);
	free(receive6_4flux2[n]);
	free(receive6_6flux2[n]);
	free(receive6_8flux2[n]);
	#endif
	free(send1_flux[n]);
	free(send2_flux[n]);
	free(send3_flux[n]);
	free(send4_flux[n]);
	#if(N3G>0)
	free(send5_flux[n]);
	free(send6_flux[n]);
	#endif
	free(receive1_E[n]);
	free(receive2_E[n]);
	free(receive3_E[n]);
	free(receive4_E[n]);
	free(receive5_E[n]);
	free(receive6_E[n]);
	free(receive1_3E[n]);
	free(receive1_4E[n]);
	free(receive1_7E[n]);
	free(receive1_8E[n]);
	free(receive2_1E[n]);
	free(receive2_2E[n]);
	free(receive2_3E[n]);
	free(receive2_4E[n]);
	free(receive3_1E[n]);
	free(receive3_2E[n]);
	free(receive3_5E[n]);
	free(receive3_6E[n]);
	free(receive4_5E[n]);
	free(receive4_6E[n]);
	free(receive4_7E[n]);
	free(receive4_8E[n]);
	#if(N3G>0)
	free(receive5_1E[n]);
	free(receive5_3E[n]);
	free(receive5_5E[n]);
	free(receive5_7E[n]);
	free(receive6_2E[n]);
	free(receive6_4E[n]);
	free(receive6_6E[n]);
	free(receive6_8E[n]);
	#endif
	free(receive1_E1[n]);
	free(receive2_E1[n]);
	free(receive3_E1[n]);
	free(receive4_E1[n]);
	free(receive5_E1[n]);
	free(receive6_E1[n]);
	free(receive1_3E1[n]);
	free(receive1_4E1[n]);
	free(receive1_7E1[n]);
	free(receive1_8E1[n]);
	free(receive2_1E1[n]);
	free(receive2_2E1[n]);
	free(receive2_3E1[n]);
	free(receive2_4E1[n]);
	free(receive3_1E1[n]);
	free(receive3_2E1[n]);
	free(receive3_5E1[n]);
	free(receive3_6E1[n]);
	free(receive4_5E1[n]);
	free(receive4_6E1[n]);
	free(receive4_7E1[n]);
	free(receive4_8E1[n]);
	#if(N3G>0)
	free(receive5_1E1[n]);
	free(receive5_3E1[n]);
	free(receive5_5E1[n]);
	free(receive5_7E1[n]);
	free(receive6_2E1[n]);
	free(receive6_4E1[n]);
	free(receive6_6E1[n]);
	free(receive6_8E1[n]);
	#endif
	free(receive1_3E2[n]);
	free(receive1_4E2[n]);
	free(receive1_7E2[n]);
	free(receive1_8E2[n]);
	free(receive2_1E2[n]);
	free(receive2_2E2[n]);
	free(receive2_3E2[n]);
	free(receive2_4E2[n]);
	free(receive3_1E2[n]);
	free(receive3_2E2[n]);
	free(receive3_5E2[n]);
	free(receive3_6E2[n]);
	free(receive4_5E2[n]);
	free(receive4_6E2[n]);
	free(receive4_7E2[n]);
	free(receive4_8E2[n]);
	#if(N3G>0)
	free(receive5_1E2[n]);
	free(receive5_3E2[n]);
	free(receive5_5E2[n]);
	free(receive5_7E2[n]);
	free(receive6_2E2[n]);
	free(receive6_4E2[n]);
	free(receive6_6E2[n]);
	free(receive6_8E2[n]);
	#endif
	free(send1_E[n]);
	free(send2_E[n]);
	free(send3_E[n]);
	free(send4_E[n]);
	#if(N3G>0)
	free(send5_E[n]);
	free(send6_E[n]);
	#endif
	free(send_E3_corn1[n]);
	free(send_E3_corn2[n]);
	free(send_E3_corn3[n]);
	free(send_E3_corn4[n]);
	#if(N3G>0)
	free(send_E2_corn5[n]);
	free(send_E2_corn6[n]);
	free(send_E2_corn7[n]);
	free(send_E2_corn8[n]);
	free(send_E1_corn9[n]);
	free(send_E1_corn10[n]);
	free(send_E1_corn11[n]);
	free(send_E1_corn12[n]);
	#endif
	free(receive_E3_corn1[n]);
	free(receive_E3_corn2[n]);
	free(receive_E3_corn3[n]);
	free(receive_E3_corn4[n]);
	#if(N3G>0)
	free(receive_E2_corn5[n]);
	free(receive_E2_corn6[n]);
	free(receive_E2_corn7[n]);
	free(receive_E2_corn8[n]);
	free(receive_E1_corn9[n]);
	free(receive_E1_corn10[n]);
	free(receive_E1_corn11[n]);
	free(receive_E1_corn12[n]);
	#endif
	free(receive_E3_corn1_1[n]);
	free(receive_E3_corn2_1[n]);
	free(receive_E3_corn3_1[n]);
	free(receive_E3_corn4_1[n]);
	#if(N3G>0)
	free(receive_E2_corn5_1[n]);
	free(receive_E2_corn6_1[n]);
	free(receive_E2_corn7_1[n]);
	free(receive_E2_corn8_1[n]);
	free(receive_E1_corn9_1[n]);
	free(receive_E1_corn10_1[n]);
	free(receive_E1_corn11_1[n]);
	free(receive_E1_corn12_1[n]);
	#endif
	free(receive_E3_corn1_2[n]);
	free(receive_E3_corn2_2[n]);
	free(receive_E3_corn3_2[n]);
	free(receive_E3_corn4_2[n]);
	#if(N3G>0)
	free(receive_E2_corn5_2[n]);
	free(receive_E2_corn6_2[n]);
	free(receive_E2_corn7_2[n]);
	free(receive_E2_corn8_2[n]);
	free(receive_E1_corn9_2[n]);
	free(receive_E1_corn10_2[n]);
	free(receive_E1_corn11_2[n]);
	free(receive_E1_corn12_2[n]);
	#endif
	free(tempreceive_E3_corn1[n]);
	free(tempreceive_E3_corn2[n]);
	free(tempreceive_E3_corn3[n]);
	free(tempreceive_E3_corn4[n]);
	#if(N3G>0)
	free(tempreceive_E2_corn5[n]);
	free(tempreceive_E2_corn6[n]);
	free(tempreceive_E2_corn7[n]);
	free(tempreceive_E2_corn8[n]);
	free(tempreceive_E1_corn9[n]);
	free(tempreceive_E1_corn10[n]);
	free(tempreceive_E1_corn11[n]);
	free(tempreceive_E1_corn12[n]);
	#endif
	free(tempreceive_E3_corn1_1[n]);
	free(tempreceive_E3_corn2_1[n]);
	free(tempreceive_E3_corn3_1[n]);
	free(tempreceive_E3_corn4_1[n]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_1[n]);
	free(tempreceive_E2_corn6_1[n]);
	free(tempreceive_E2_corn7_1[n]);
	free(tempreceive_E2_corn8_1[n]);
	free(tempreceive_E1_corn9_1[n]);
	free(tempreceive_E1_corn10_1[n]);
	free(tempreceive_E1_corn11_1[n]);
	free(tempreceive_E1_corn12_1[n]);
	#endif
	free(tempreceive_E3_corn1_2[n]);
	free(tempreceive_E3_corn2_2[n]);
	free(tempreceive_E3_corn3_2[n]);
	free(tempreceive_E3_corn4_2[n]);
	#if(N3G>0)
	free(tempreceive_E2_corn5_2[n]);
	free(tempreceive_E2_corn6_2[n]);
	free(tempreceive_E2_corn7_2[n]);
	free(tempreceive_E2_corn8_2[n]);
	free(tempreceive_E1_corn9_2[n]);
	free(tempreceive_E1_corn10_2[n]);
	free(tempreceive_E1_corn11_2[n]);
	free(tempreceive_E1_corn12_2[n]);
	#endif
	free(receive_E3_corn1_12[n]);
	free(receive_E3_corn2_12[n]);
	free(receive_E3_corn3_12[n]);
	free(receive_E3_corn4_12[n]);
	#if(N3G>0)
	free(receive_E2_corn5_12[n]);
	free(receive_E2_corn6_12[n]);
	free(receive_E2_corn7_12[n]);
	free(receive_E2_corn8_12[n]);
	free(receive_E1_corn9_12[n]);
	free(receive_E1_corn10_12[n]);
	free(receive_E1_corn11_12[n]);
	free(receive_E1_corn12_12[n]);
	#endif
	free(receive_E3_corn1_22[n]);
	free(receive_E3_corn2_22[n]);
	free(receive_E3_corn3_22[n]);
	free(receive_E3_corn4_22[n]);
	#if(N3G>0)
	free(receive_E2_corn5_22[n]);
	free(receive_E2_corn6_22[n]);
	free(receive_E2_corn7_22[n]);
	free(receive_E2_corn8_22[n]);
	free(receive_E1_corn9_22[n]);
	free(receive_E1_corn10_22[n]);
	free(receive_E1_corn11_22[n]);
	free(receive_E1_corn12_22[n]);
	#endif
	#endif
	//#endif
	free(Katm[n]);
	free(array[n]);
	free(array_rdump[n]);
	free(array_diag[n]);
}

void set_arrays(int n)
{
	int i, j, z, k;
	array[n] = (float *)calloc(9 * BS_1*BS_2*BS_3, sizeof(float));
	array_rdump[n] = (double *)calloc((NPR+NDIM) * (BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G), sizeof(double));
	array_diag[n] = (float *)calloc(4 * BS_1*BS_2*BS_3, sizeof(float));

	Katm[n] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	pbound[n] = (double(*)[NPR][N_POINTS])calloc((N2 + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NPR][N_POINTS]));
	p[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	V[n] = (double(*)[6])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[6]));
	E_avg[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_x[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_x[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new_x[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new_x[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_y[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_y[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new_y[n][0] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	E_avg_new_y[n][1] = (double(*))calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#if(STAGGERED)
	ps[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NDIM]));
	psh[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NDIM]));
	dE[n] = (double(*)[2][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[2][NDIM][NDIM]));
	#endif
	E_corn[n] = (double(*)[NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NDIM]));
	dq[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	//#if(GPU_BENCHMARK || GPU_DEBUG || CPU_OPENMP)
	F1[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	F2[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	F3[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	ph[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	//#endif
	#if(STAGGERED)
	stor1[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	stor2[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPR]));
	#endif
	pflag[n] = (int(*))calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(int));
	failimage[n] = (int(*)[NFAIL])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(int[NFAIL]));
	#if(!NONSYMMETRIC)
	conn[n] = (double(*)[NDIM][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G),sizeof(double[NDIM][NDIM][NDIM]));
	gcov[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) ,sizeof(double[NPG][NDIM][NDIM]));
	gcon[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G),sizeof(double[NPG][NDIM][NDIM]));
	gdet[n] = (double(*)[NPG])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G),sizeof(double[NPG]));
	#else
	conn[n] = (double(*)[NDIM][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NDIM][NDIM][NDIM]));
	gcov[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NPG][NDIM][NDIM]));
	gcon[n] = (double(*)[NPG][NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPG][NDIM][NDIM]));
	gdet[n] = (double(*)[NPG])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G),sizeof(double[NPG]));
	#endif
	#if(ZIRI_DUMP)
	dump_buffer[n] = (double(*))calloc(N1_GPU[n] * N2_GPU[n] * N3_GPU[n] * 13 ,sizeof(double));
	dxdxp_z[n] = (double(*)[NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NDIM][NDIM]));
	dxpdx_z[n] = (double(*)[NDIM][NDIM])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double[NDIM][NDIM]));
	#endif
	#if (ELLIPTICAL2)
	dU_s[n] = (double(*)[NPR])calloc((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G),sizeof(double[NPR]));
	#endif
	send[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)  ,sizeof(double));
	//#if(CPU_OPENMP|| PINNED==0)
	receive1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	tempreceive1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	tempreceive2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	tempreceive3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	tempreceive4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	tempreceive5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	tempreceive6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	send1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G),sizeof(double));
	send2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	send5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) * (N1_GPU[n] + 2 * N1G),sizeof(double));
	send6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) * (N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	#if(N_LEVELS>1)
	send1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_3[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_4[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_7[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive1_8[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive2_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_1[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_2[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_5[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive3_6[n] = (double *)calloc(NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	send4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	send4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	receive4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	tempreceive4_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G), sizeof(double));
	#if(N3G>0)
	send5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_1[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_3[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_5[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive5_7[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	send6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	send6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	receive6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_2[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_4[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_6[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	tempreceive6_8[n] = (double *)calloc(NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G), sizeof(double));
	#endif
	#endif
	receive1_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	send1_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G),sizeof(double));
	send2_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send3_fine[n] = (double *)calloc(NG * NPR*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send4_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	send5_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G),sizeof(double));
	send6_fine[n] = (double *)calloc(NG * NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	receive1_3fine[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_4fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_7fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive1_8fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_1fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_2fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_3fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive2_4fine[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_1fine[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_2fine[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_5fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive3_6fine[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_5fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_6fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_7fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	receive4_8fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G), sizeof(double));
	#if(N3G>0)
	receive5_1fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_3fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_5fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive5_7fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_2fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_4fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_6fine[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	receive6_8fine[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G), sizeof(double));
	#endif
	#if(CPU_OPENMP)
	receive1_flux[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3_flux[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_flux[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	receive1_flux1[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3_flux1[n] = (double *)calloc(NPR* (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_flux1[n] = (double *)calloc(NPR* (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6_flux1[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive1_3flux[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4flux[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1flux[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2flux[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6flux[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6flux[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8flux[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	receive1_3flux1[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4flux1[n] = (double *)calloc(NPR * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1flux1[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2flux1[n] = (double *)calloc(NPR * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6flux1[n] = (double *)calloc(NPR* (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6flux1[n] = (double *)calloc(NPR* (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8flux1[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	receive1_3flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6flux2[n] = (double *)calloc(NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8flux2[n] = (double *)calloc(NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	receive1_3E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6E[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8E[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	receive1_3E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6E1[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8E1[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	receive1_3E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_4E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_7E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive1_8E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_1E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_2E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_3E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive2_4E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_1E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_2E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_5E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive3_6E2[n] = (double *)calloc(2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_5E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_6E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_7E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	receive4_8E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_1E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_3E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_5E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive5_7E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_2E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_4E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_6E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	receive6_8E2[n] = (double *)calloc(2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G) *(N1_GPU[n] / (1 + REF_1) + 2 * N1G),sizeof(double));
	#endif
	#endif
	send1_flux[n] = (double *)calloc(NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G),sizeof(double));
	send2_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send3_flux[n] = (double *)calloc(NPR*(N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send4_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	send5_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G),sizeof(double));
	send6_flux[n] = (double *)calloc(NPR*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	receive1_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	receive1_E1[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive2_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive3_E1[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	receive4_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	receive5_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	receive6_E1[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	send1_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G),sizeof(double));
	send2_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send3_E[n] = (double *)calloc(2 * (N1_GPU[n] + 2 * N1G) *(N3_GPU[n] + 2 * N3G),sizeof(double));
	send4_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) ,sizeof(double));
	#if(N3G>0)
	send5_E[n] = (double *)calloc(2 * (N2_GPU[n] + 2 * N2G) *(N1_GPU[n] + 2 * N1G),sizeof(double));
	send6_E[n] = (double *)calloc(2*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) ,sizeof(double));
	#endif
	
	send_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	send_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	send_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	send_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	send_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	send_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	send_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	send_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	send_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	send_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	send_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	send_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	receive_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	receive_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	tempreceive_E3_corn1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn3[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn4[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn6[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn7[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn8[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E1_corn9[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn10[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn11[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	#if(N_LEVELS>1)
	receive_E3_corn1_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn2_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn3_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn4_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn6_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn7_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn8_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E1_corn9_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn10_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn11_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn12_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	receive_E3_corn1_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn2_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn3_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn4_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn6_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn7_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn8_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E1_corn9_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn10_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn11_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn12_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	tempreceive_E3_corn1_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn2_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn3_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn4_1[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn6_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn7_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn8_1[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E1_corn9_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn10_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn11_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn12_1[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	tempreceive_E3_corn1_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn2_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn3_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	tempreceive_E3_corn4_2[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	tempreceive_E2_corn5_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn6_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn7_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E2_corn8_2[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	tempreceive_E1_corn9_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn10_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn11_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	tempreceive_E1_corn12_2[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	receive_E3_corn1_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn2_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn3_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn4_12[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn6_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn7_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn8_12[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E1_corn9_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn10_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn11_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn12_12[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	receive_E3_corn1_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn2_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn3_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	receive_E3_corn4_22[n] = (double *)calloc((N3_GPU[n] + 2 * N3G),sizeof(double));
	#if(N3G>0)
	receive_E2_corn5_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn6_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn7_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E2_corn8_22[n] = (double *)calloc((N2_GPU[n] + 2 * N2G),sizeof(double));
	receive_E1_corn9_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn10_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn11_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	receive_E1_corn12_22[n] = (double *)calloc((N1_GPU[n] + 2 * N1G),sizeof(double));
	#endif
	#endif
	#endif
	//#endif
}

int index(int n, int i, int j, int z)
{
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(N3_GPU[n] + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
}
int index2(int n, int i, int j, int z)
{
	#if(!NONSYMMETRIC)
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G) + ((j - N2_GPU_offset[n]) + N2G));
	#else
	return(((i - N1_GPU_offset[n]) + N1G)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) + ((j - N2_GPU_offset[n]) + N2G)*(N3_GPU[n] + 2 * N3G) + ((z - N3_GPU_offset[n]) + N3G));
	#endif
}
int index3(int i, int j)
{
	return((i + N1G)*(N2 + 2 * N2G) + (j + N2G));
}

/*****************************************************************/
/*****************************************************************
  set_grid():
  ----------

       -- calculates all grid functions that remain constant 
          over time, such as the metric (gcov), inverse metric 
          (gcon), connection coefficients (conn), and sqrt of 
          the metric's determinant (gdet).

 *****************************************************************/
void set_grid(int n)
{
	int i,j,z,k,i1,j1,z1 ;
	double r, th, phi;
	struct of_geom geom ;

	/* set up boundaries, steps in coordinate grid */
	set_points(n) ;
	dV = dx[n][1] * dx[n][2] * dx[n][3];
	double X[NDIM];

	double temp = a;
	#pragma omp parallel private(X,i,j,z,k,geom, i1,j1,z1,r,th,phi,a)
	{
		DLOOPA X[j] = 0.;
		#pragma omp for collapse(2) schedule(dynamic)
		#if(!NONSYMMETRIC)
		ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]) {
		#else
		ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		#endif
			if (j<0 || j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND) a = -temp;
			else a = temp;
			
			/* zone-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND == -1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n,i, j, z, CENT, X);
			gcov_func(X, gcov[n][index2(n, i, j, z)][CENT]);
			gdet[n][index2(n, i, j, z)][CENT] = gdet_func(gcov[n][index2(n, i, j, z)][CENT]);
			if (j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])-1 && TRANS_BOUND == 1)gdet[n][index2(n, i, j, z)][CENT] *= 1.0;
			gcon_func(gcov[n][index2(n, i, j, z)][CENT], gcon[n][index2(n, i, j, z)][CENT]);
			get_geometry(n, i, j, z, CENT, &geom);
			conn_func(X, &geom, conn[n][index2(n, i, j, z)]);
			if ((j == -1 || j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])) && (TRANS_BOUND==1)){
				//for (i1 = 0; i1 < NDIM; i1++)for (j1 = 0; j1 < NDIM; j1++)for (z1 = 0; z1 < NDIM; z1++)conn[n][index2(n, i, j, z)][i1][j1][z1] = 0.;
			}

			/* corner-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n, i, j, z, FACE1, X);
			gcov_func(X, gcov[n][index2(n, i, j, z)][CORN]);
			gdet[n][index2(n, i, j, z)][CORN] = gdet_func(gcov[n][index2(n, i, j, z)][CORN]);
			gcon_func(gcov[n][index2(n, i, j, z)][CORN], gcon[n][index2(n, i, j, z)][CORN]);

			/* r-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, CORN, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, CORN, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, CORN, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, CORN, X);
			else coord(n, i, j, z, FACE1, X);
			gcov_func(X, gcov[n][index2(n, i, j, z)][FACE1]);
			gdet[n][index2(n, i, j, z)][FACE1] = gdet_func(gcov[n][index2(n, i, j, z)][FACE1]);
			gcon_func(gcov[n][index2(n, i, j, z)][FACE1], gcon[n][index2(n, i, j, z)][FACE1]);
			
			/* phi-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==-1) coord(n, i, 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) + 1, z, FACE2, X);
			else coord(n, i, j, z, FACE3, X);
			gcov_func(X, gcov[n][index2(n, i, j, z)][FACE3]);
			gdet[n][index2(n, i, j, z)][FACE3] = gdet_func(gcov[n][index2(n, i, j, z)][FACE3]);
			gcon_func(gcov[n][index2(n, i, j, z)][FACE3], gcon[n][index2(n, i, j, z)][FACE3]);

			/* theta-face-centered */
			if (j == -1 && TRANS_BOUND==-1)coord(n, i, -1, z, FACE2, X);
			else if (j == 0 && TRANS_BOUND==1){
				//coord(n, i, 1, z, FACE2, X);
				a = 0. ;
				coord(n, i, j, z, FACE2, X);

			}
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 && TRANS_BOUND==-1) coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
			else if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) && TRANS_BOUND==1){
				//coord(n, i, N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z, FACE2, X);
				coord(n, i, j, z, FACE2, X);
				a = 0.;
			}
			else coord(n, i, j, z, FACE2, X);
			gcov_func(X, gcov[n][index2(n, i, j, z)][FACE2]);
			if ((j == 0  || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])) && TRANS_BOUND==1){
				//gcov[n][index2(n, i, j, z)][FACE2][2][1] = 0.;
				//gcov[n][index2(n, i, j, z)][FACE2][1][2] = 0.;
				//gcov[n][index2(n, i, j, z)][FACE2][2][3] = 0.;
				//gcov[n][index2(n, i, j, z)][FACE2][3][2] = 0.;
				//gcov[n][index2(n, i, j, z)][FACE2][1][3] = 0.;
				//gcov[n][index2(n, i, j, z)][FACE2][3][1] = 0.;
			}
			/*if (j == -1 || j == 0 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL]) - 1 || j == N2*pow(1 + REF_2, block[n][AMR_LEVEL])){
				double dxdxp[NDIM][NDIM], dxdxp_inv[NDIM][NDIM];
				int I, J, K, L;
				dxdxp_func(X, dxdxp);
				invert_matrix(dxdxp, dxdxp_inv);
				for (I = 0; I<NDIM; I++){
					for (J = 0; J<NDIM; J++){
						gcon[n][index2(n, i, j, z)][FACE2][I][J] = 0.;
						for (K = 0; K<NDIM; K++) {
							for (L = 0; L<NDIM; L++){
								gcon[n][index2(n, i, j, z)][FACE2][I][J] += gcov[n][index2(n, i, j, z)][FACE2][K][L] * dxdxp_inv[K][I] * dxdxp_inv[L][J];
							}
						}
					}
				}
				coord(n, i, j, z, FACE2, X);
				dxdxp_func(X, dxdxp);
				for (I = 0; I < NDIM; I++){
					for (J = 0; J < NDIM; J++){
						gcov[n][index2(n, i, j, z)][FACE2][I][J] = 0.;
						for (K = 0; K < NDIM; K++) {
							for (L = 0; L < NDIM; L++){
								gcov[n][index2(n, i, j, z)][FACE2][I][J] += gcon[n][index2(n, i, j, z)][FACE2][K][L] * dxdxp[K][I] * dxdxp[L][J];
							}
						}
					}
				}
			}*/
			gdet[n][index2(n, i, j, z)][FACE2] = gdet_func(gcov[n][index2(n, i, j, z)][FACE2]);
			gcon_func(gcov[n][index2(n, i, j, z)][FACE2], gcon[n][index2(n, i, j, z)][FACE2]);	
		}
	}

	#if(LEER)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		//Set temporary array with r, th, phi distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		coord(n, i, j, z, FACE1, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][0] = r;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][3] = r;

		coord(n, i, j, z, FACE2, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][1] = th;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][4] = th;

		coord(n, i, j, z, FACE3, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][2] = phi;
		
		coord(n, i, j, z, CENT, X);
		bl_coord(X, &r, &th, &phi);
		dq[n][index(n, i, j, z)][5] = phi;

		for (k = 0; k < 6; k++) V[n][index(n, i, j, z)][k] = 0.0;
	}
	ZSLOOP3D(-N1G + N1_GPU_offset[n],-N1G + N1_GPU_offset[n], -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		V[n][index(n, i, j, z)][3] = V[n][index(n, i, j, z)][0] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE1][1][1]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], -N2G + N2_GPU_offset[n], -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		V[n][index(n, i, j, z)][4] = V[n][index(n, i, j, z)][1] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE2][2][2]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1 + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], -N3G + N3_GPU_offset[n]) {
		V[n][index(n, i, j, z)][5] = V[n][index(n, i, j, z)][2] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE3][3][3]);//(r*sin(th)*dphi)^2
	}

	ZSLOOP3D(-D1+ N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -D2 + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -D3 + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		//Calculate distances between pixels in x1,x2,x3-->0,1,2 at the faces of the cell and x1,x2,x3-->3,4,5 at the cell centres
		V[n][index(n, i, j, z)][0] = V[n][index(n, i - D1, j, z)][3] + 0.5*sqrt(gcov[n][index2(n, i - D1, j, z)][CENT][1][1]);
		V[n][index(n, i, j, z)][3] = V[n][index(n, i, j, z)][0] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE1][1][1]);
		
		V[n][index(n, i, j, z)][1] = V[n][index(n, i, j - D2, z)][4] + 0.5*sqrt(gcov[n][index2(n, i, j - D2, z)][CENT][2][2]);
		V[n][index(n, i, j, z)][4] = V[n][index(n, i, j, z)][1] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE2][2][2]);

		V[n][index(n, i, j, z)][2] = V[n][index(n, i, j, z - D3)][5] + 0.5*sqrt(gcov[n][index2(n, i, j, z - D3)][CENT][3][3]);
		V[n][index(n, i, j, z)][5] = V[n][index(n, i, j, z)][2] + 0.5*sqrt(gcov[n][index2(n, i, j, z)][FACE3][3][3]);
	}
	#endif

	a=temp;

	#if ZIRI_DUMP
	ZSLOOP3D(-N1G + N1_GPU_offset[n], N1_GPU[n] + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G) {
		coord(n,i, j, z, CENT, X);
		dxdxp_func(X, dxdxp_z[n][index(n ,i,j,z)]);
		//invert_matrix(dxdxp_z[n][index(n ,i,j,z)], dxpdx_z[n][index(n ,i,j,z)]);
	}
	#endif

	/* done! */
}


/*This function initialises the MPI structure. It divides the grid(N1, N2, N3) among the MPI processes.
The host node is node 0 by default.*/

void MPI_initialize(int argc, char *argv[])
{

#if (MPI_enable)
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int i, j, z, len, dim, corn, rankloop;
	int dims[3], periods[3], coords[3];

	int rdma_direct = getenv("MPICH_RDMA_ENABLED_CUDA") == NULL ? 0 : atoi(getenv("MPICH_RDMA_ENABLED_CUDA"));
	if (rdma_direct != 1){
		printf("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
	}

	/*Get basic initialisation*/
	local_rank = 0;// atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
	cudaSetDevice(local_rank%N_GPU);

	rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &i);
	//rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		fprintf(stderr, "Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Get_processor_name(hostname, &len);

	/*Handy for debugging, you don't have to change this value everytime if you want to test MPI vs non-MPI*/

	/*if (numtasks != 1){
		n_columns = MPI_columns;
		n_rows = MPI_rows;
		n_stacks = MPI_stacks;
		}
		else{
		n_rows = 1;
		n_columns = 1;
		n_stacks = 1;
		}
		dims[0] = n_columns;
		dims[1] = n_rows;
		dims[2] = n_stacks;
		periods[0] = 0;
		periods[1] = 0;
		periods[2] = 1;

		//Initialize cartesian communicator
		MPI_Cart_create(MPI_COMM_WORLD, 3, dims , periods , 1, &mpi_cartcomm);
		MPI_Comm_rank(mpi_cartcomm, &rank);
		MPI_Cart_coords(mpi_cartcomm,rank, 3,  coords);
		//MPI_Errhandler_set(mpi_cartcomm, MPI_ERRORS_RETURN);
		//N3_GPU[n_ord[n]] = coords[2];
		//N2_GPU[n_ord[n]] = coords[1];
		//N1_GPU[n_ord[n]] = coords[0];

		for (dim = 0; dim < 3; dim++) {
		MPI_Cart_shift(mpi_cartcomm, dim , 1, &mpi_nbrs[dim+1][0], &mpi_nbrs[dim+1][1]);
		}*/
	dims[0] = NB_1;
	dims[1] = NB_2;
	dims[2] = NB_3;
	periods[0] = 0;
	periods[1] = 0;
	periods[2] = 1;

	//Initialize cartesian communicator
	//if(NB_1*NB_2*NB_3==numtasks){
	//	MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &mpi_cartcomm);
	//	MPI_Comm_rank(mpi_cartcomm, &rank);
	//	fprintf(stderr,"Using cartesian communicator for MPI /n");
	//}
	//else{
		//Initialize normal communicator
		mpi_cartcomm = MPI_COMM_WORLD;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &mpi_self);

	//}
	
	//if (MPI_THREAD_MULTIPLE != i && rank==0) fprintf(stderr, "MPI library has unsufficient threading support\n");

	/*Give notice if programmer makes error in selecting the MPI_rows and MPI_column values in dec.h*/
	//if (rank == 0 && n_rows*n_columns != numtasks){
	//	fprintf(stderr, "Error: Number of tasks is not equal to rows*columns! \n");
	//}

	/*Give basic diagnostics*/
	if (rank == 0){
		fprintf(stderr, "Number of MPI tasks: %d \nRunning on: %s\n", numtasks, hostname);
		//fprintf(stderr, "MPI geometry(columns, rows, stacks) : (%d, %d, %d)\n", n_columns, n_rows, n_stacks);
	}
	#else
	numtasks = 1;
	rank = 0;
	N1_GPU_offset[n_ord[n]] = 0;
	N1_GPU[n_ord[n]] = N1;
	N2_GPU_offset[n_ord[n]] = 0;
	N2_GPU[n_ord[n]] = N2;
	N3_GPU_offset[n_ord[n]] = 0;
	N3_GPU[n_ord[n]] = N3;
	n_rows = 1;
	n_columns = 1;
	n_stacks = 1;
	#endif
}

void mpi_synch(void){
	int i;
	for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
		if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
		break;
		}
	}
}