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


/* restart functions; restart_init and restart_dump */

#include "decs.h"

/***********************************************************************/
/***********************************************************************
  restart_write():
     -- writes current state of primitive variables to the 
        checkpointing/restart file. 
     -- uses ASCII text format ;
     -- when changing this routine, be sure to make analogous changes 
        in restart_read();
************************************************************************/
void restart_write()
{
  FILE *fp ;
  int idum,i,j,z, k, l ;
  int int_size = sizeof(int);
  int double_size = sizeof(double);
  MPI_Barrier(MPI_COMM_WORLD);
  for (l = 0; l < numtasks; l++){
	  if (rank == l){
		  if (rank == 0){
			  if (rdump_cnt % 2 == 0) {
				  fp = fopen("dumps/rdump0.bin", "wb");
				  fprintf(stderr, "RESTART  file=dumps/rdump0\n");
			  }
			  else {
				  fp = fopen("dumps/rdump1.bin", "wb");
				  fprintf(stderr, "RESTART file=dumps/rdump1\n");
			  }
			  if (fp == NULL) {
				  fprintf(stderr, "Cannot open restart file\n");
				  exit(2);
			  }
		  }
		  else{
			  if (rdump_cnt % 2 == 0) {
				  fp = fopen("dumps/rdump0.bin", "ab");
			  }
			  else {
				  fp = fopen("dumps/rdump1.bin", "ab");
			  }
			  if (fp == NULL) {
				  fprintf(stderr, "Cannot open restart file\n");
				  exit(2);
			  }
		  }
		  /*************************************************************
		  Write the header of the restart file:
		  *************************************************************/
		  if (rank == 0){
			  int N1_print = N1;
			  int N2_print = N2;
			  int N3_print = N3;
			  fwrite(&N1_print, int_size, 1, fp);
			  fwrite(&N2_print, int_size, 1, fp);
			  fwrite(&N3_print, int_size, 1, fp);
			  fwrite(&n_rows, int_size, 1, fp);
			  fwrite(&n_columns, int_size, 1, fp);
			  fwrite(&n_stacks, int_size, 1, fp);
			  fwrite(&t, double_size, 1, fp);
			  fwrite(&tf, double_size, 1, fp);
			  fwrite(&fractheta, double_size, 1, fp);
			  fwrite(&nstep, int_size, 1, fp);
			  fwrite(&a, double_size, 1, fp);
			  fwrite(&gam, double_size, 1, fp);
			  fwrite(&cour, double_size, 1, fp);
			  fwrite(&DTd, double_size, 1, fp);
			  fwrite(&DTl, double_size, 1, fp);
			  fwrite(&DTi, double_size, 1, fp);
			  fwrite(&DTr, int_size, 1, fp);
			  fwrite(&dump_cnt, int_size, 1, fp);
			  fwrite(&image_cnt, int_size, 1, fp);
			  fwrite(&rdump_cnt, int_size, 1, fp);
			  fwrite(&dt, double_size, 1, fp);
			  fwrite(&lim, int_size, 1, fp);
			  fwrite(&failed, int_size, 1, fp);
			  fwrite(&Rin, double_size, 1, fp);
			  fwrite(&Rout, double_size, 1, fp);
			  fwrite(&hslope, double_size, 1, fp);
			  fwrite(&R0, double_size, 1, fp);
		  }

		  /*************************************************************
		  Write the body of the restart file:
		  *************************************************************/
		  ZSLOOP3D(-N1G + N1_MPI_offset, N1_MPI_offset + N1_MPI - 1+N1G, -N2G + N2_MPI_offset, N2_MPI_offset + N2_MPI - 1+N2G, -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI - 1+N3G) {
			  PLOOP fwrite(&(p[index(i, j, z)][k]), double_size, 1, fp);
		  }
		  fclose(fp);
		  rdump_cnt++;
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
  }
  return;
}

/***********************************************************************/
/***********************************************************************
  restart_init():
     -- main driver for setting initial conditions from a checkpoint 
        or restart file. 
     -- determines if there are any restart files to use and then 
         lets the  user choose if there are more than one file. 
     -- then calls initializes run with restart data;
************************************************************************/
int restart_init()
{
  FILE *fp, *fp1, *fp0 ;
  char ans[100] ;
  int i,j,k, l, nofile=0;
  double r, th, phi;
  double trash;

  /********************************************************************
   Check to see which restart files exist. 
   Use the only one that exists, else prompt user to decide 
     which one to use if we have a choice : 
  ********************************************************************/

    fp0 = fopen("dumps/rdump0.bin", "rb");
	fp1 = fopen("dumps/rdump1.bin", "rb");
	#if (RESTART==1)
    fp0=NULL;
	#elif (RESTART==0)
    fp1 = NULL;
    #endif

  if ((fp0 == NULL) && (fp1 == NULL)) {
	  if (rank == 0){
		  fprintf(stderr, "No restart file\n");
	  }
	  nofile = 1;
  }
  for (l = 0; l < numtasks; l++){
	  if (rank == l){
		  if(nofile==0) {
			  if (rank == 0){
				  fprintf(stderr, "\nRestart file exists! \n");
			  }
			  if (fp0 == NULL) {
				  if (rank == 0){
					  fprintf(stderr, "Using dumps/rdump1 ... \n");
				  }
				  fp = fopen("dumps/rdump1.bin", "rb");
			  }
			  else if (fp1 == NULL) {
				  if (rank == 0){
					  fprintf(stderr, "Using dumps/rdump0 ... \n");
				  }
				  fp = fopen("dumps/rdump0.bin", "rb");;
			  }
			  else {
				  if (rank == 0){
					  fprintf(stderr, "Use dumps/rdump0 (0) or dumps/rdump1 (1)?   [0|1]  \n");
				  }
				  fscanf(stdin, "%s", ans);
				  if (strncmp(ans, "0", 1) == 0) {
					  fp = fopen("dumps/rdump0.bin", "rb");
				  }
				  else{
					  fp = fopen("dumps/rdump1.bin", "rb");
				  }
			  }
			 
			  /********************************************************************
			   Now that we know we are restarting from a checkpoint file, then
			   we need to read in data, assign grid functions and define the grid:
			   ********************************************************************/
			  /* set up global arrays */
			  set_arrays();
			  
			  /*Read in file*/
			  restart_read(fp);
			  fclose(fp);

			  /* set metric functions */
			  set_grid();
			  
			  #if( DO_FONT_FIX ) 
			  set_Katm();
			  #endif 
	
			  /***********************************************************************
				Make any changes to parameters in restart file  here:
				e.g., cour = 0.4 , change in limiter...
				************************************************************************/
			  //lim = MC ;
			  //cour = 0.9 ;
			  //lim = VANL ;
			  //tf = 4000. ;

			  if (rank == 0){
				  fprintf(stderr, "done with restart init.\n");
			  }
		  }
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	
	  if (nofile == 1){
		  return(0);
	  }
  }

  /* bound */
  bound_prim(p,1);

 /* done! */
  return(1) ;
}

/***********************************************************************/
/***********************************************************************
  restart_read():
     -- reads in data from the restart file, which is specified in 
         restart_init() but is usually named "dumps/rdump[0,1]" 
************************************************************************/
void restart_read(FILE *fp)
{
  int idum,i,j,z,k,l,point;
  int int_size = sizeof(int);
  int double_size = sizeof(double);
  double trash;

  /*************************************************************
	  READ the header of the restart file: 
  *************************************************************/
  fread(&idum, int_size, 1,fp );
  if(idum != N1 && rank==0 ) {
    fprintf(stderr,"Error reading restart file; N1 differs. Select N1=%d. \n", idum-ibound) ;
    exit(3) ;
  }
  fread(&idum, int_size, 1, fp);
  if(idum != N2 && rank==0) {
	  fprintf(stderr, "Error reading restart file. N2 differs. Select N2=%d. \n", idum - jbound);
    exit(4) ;
  }
  fread(&idum, int_size, 1, fp);
  if (idum != N3 && rank == 0) {
	  fprintf(stderr, "Error reading restart file. N3 differs. Select N3=%d. \n", idum - zbound);
	  exit(5);
  }
  fread(&idum, int_size, 1, fp);
  if (idum != n_rows && rank==0){
	  fprintf(stderr, "Error reading restart file, n_rows differs!\n");
	  exit(6);
  }
  fread(&idum, int_size, 1, fp);
  if (idum != n_columns && rank == 0){
	  fprintf(stderr, "Error reading restart file, n_columns differs!\n");
	  exit(7);
  }
  fread(&idum, int_size, 1, fp);
  if (idum != n_stacks && rank == 0){
	  fprintf(stderr, "Error reading restart file, n_stacks differs!\n");
	  exit(8);
  }
  fread(&t, double_size,1,fp );
  fread(&tf, double_size, 1, fp);
  fread(&fractheta, double_size, 1, fp);
  fread(&nstep, int_size, 1, fp);
  fread(&a, double_size, 1, fp);
  fread(&gam, double_size, 1, fp);
  fread(&cour, double_size, 1, fp);
  fread(&DTd, double_size, 1, fp);
  fread(&DTl, double_size, 1, fp);
  fread(&DTi, double_size, 1, fp);
  fread(&DTr, int_size, 1, fp);
  fread(&dump_cnt, int_size, 1, fp);
  fread(&image_cnt, int_size, 1, fp);
  fread(&rdump_cnt, int_size, 1, fp);
  fread(&dt, double_size, 1, fp);
  fread(&lim, int_size, 1, fp);
  fread(&failed, int_size, 1, fp);
  fread(&Rin, double_size, 1, fp);
  fread(&Rout, double_size, 1, fp);
  fread(&hslope, double_size, 1, fp);
  fread(&R0, double_size, 1, fp);

  /*************************************************************
	  READ the body of the restart file: 
  *************************************************************/	
  for (l = 0; l < numtasks; l++){
	  if (rank == l){
		  ZSLOOP3D(-N1G + N1_MPI_offset, N1_MPI_offset + N1_MPI - 1 + N1G, -N2G + N2_MPI_offset, N2_MPI_offset + N2_MPI - 1 + N2G,
			  -N3G + N3_MPI_offset, N3_MPI_offset + N3_MPI - 1 + N3G){
			  PLOOP fread(&(p[index(i, j, z)][k]), double_size, 1, fp);
			  /*if (boundfreeze1 || boundfreeze2){
				 for(point=0; point<N_POINTS; point++){
					 if (i == ibound+point){
						  PLOOP pbound[j][k][points] = p[ibound+point][j][k];
					  }
				  }
			  }*/
		  }
	  }
	  else{
		  ZSLOOP3D(-N1G + aN1_MPI_offset[l], aN1_MPI_offset[l] + aN1_MPI[l] - 1 + N1G, -N2G + aN2_MPI_offset[l], aN2_MPI_offset[l] + aN2_MPI[l] - 1 + N2G,
			  -N3G + aN3_MPI_offset[l], aN3_MPI_offset[l] + aN3_MPI[l] - 1 + N3G) {
			  PLOOP fread(&trash, double_size, 1, fp);
		  }
	  }
  }
  return ;
}

#undef FMT_DBL_OUT
#undef FMT_INT_OUT
