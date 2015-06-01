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

#define FLOOP for(k=0;k<B1;k++)

/* apply floors to density, internal energy */

void fixup(double((* pv)[NPR]))
{
	int i, j, z;

	// ZLOOP {
	#pragma omp parallel shared(pv) private(i,j,z)
	{
		#pragma omp for schedule(static,1)
		ZLOOP3D_MPI{
			fixup1zone(i, j, z, pv[index(i, j, z)]);
		}
	}

}

void fixup1zone( int i, int j, int z, double pv[NPR] ) 
{
  double r,th, phi, X[NDIM],uuscal,rhoscal, rhoflr,uuflr;
  double f,gamma, bsq;
  double pv_prefloor[NPR], dpv[NPR], U_prefloor[NPR], dU[NPR], U[NPR], U_ent;
  int k, flag, dofloor;
  struct of_state q;
  struct of_geom geom ;

  coord(i,j, z, CENT,X) ;
  bl_coord(X,&r,&th, &phi) ;

  rhoscal = pow(r,-POWRHO) ;
  uuscal = pow(rhoscal,gam);

  rhoflr = RHOMIN*rhoscal;
  uuflr  = UUMIN*uuscal;

  //compute the square of fluid frame magnetic field (twice magnetic pressure)
  get_geometry(i,j,CENT,&geom) ;
  bsq = bsq_calc(pv,&geom) ;
  
  //tie floors to the local values of magnetic field and internal energy density
#if(1)
  if( rhoflr < bsq / BSQORHOMAX ) rhoflr = bsq / BSQORHOMAX;
  if( uuflr < bsq / BSQOUMAX ) uuflr = bsq / BSQOUMAX;
  if( rhoflr < pv[UU] / UORHOMAX ) rhoflr = pv[UU] / UORHOMAX;
#endif

  if( rhoflr < RHOMINLIMIT ) rhoflr = RHOMINLIMIT;
  if( uuflr  < UUMINLIMIT  ) uuflr  = UUMINLIMIT;

  /* floor on density and internal energy density (momentum *not* conserved) */
#pragma simd 
  PLOOP pv_prefloor[k] = pv[k];
	if (pv[RHO] < rhoflr){
		pv[RHO] = rhoflr;
		dofloor = 1;
	}
	if (pv[UU] < uuflr){
		pv[UU] = uuflr;
		dofloor = 1;
	}

	#if( ZAMO_FLOOR )
		if (dofloor && t > 0) {
			//new way of floors according to Jon: add floors in the ZAMO frame
			//instead of fluid frame to avoid run-away

			//find the change in primitive quantities
			#pragma simd  
			for (k = 0; k < NPR; k++){
				dpv[k] = pv[k] - pv_prefloor[k];
			}

			//compute the conserved quantity associated with floor addition
			get_state(dpv, &geom, &q);
			primtoU(dpv, &q, &geom, dU);

			//compute the prefloor conserved quantity
			get_state(pv_prefloor, &geom, &q);
			primtoU(pv_prefloor, &q, &geom, U_prefloor);

			//add U_added to the current conserved quantity
			#pragma simd
			PLOOP U[k] = U_prefloor[k] + dU[k];

			//invert to obtain primitive quantity
			flag = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pv);
			if (flag){
				failimage[index(i, j, z)][0]++;
				U_ent = (geom.g *pv[0] * (gam - 1.)*pv[1] / pow(pv[0], gam)) * (q.ucon[0]);
				pflag[index(i, j, z)] = flag;
				#if( DO_FONT_FIX ) 
				pflag[index(i, j, z)] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pv, U_ent);
				if (pflag[index(i, j, z)]) {
					failimage[index(i, j, z)][1]++;
					//pflag[index(i, j, z)] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pv, U_ent);
					if (pflag[index(i, j, z)]){
						pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] = 100;
						failimage[index(i, j, z)][2]++;
					}
				}
				#else
				pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] = 100;
				#endif	
			}
		}
	#endif

  /* limit gamma wrt normal observer */

  if( gamma_calc(pv,&geom,&gamma) ) { 
    /* Treat gamma failure here as "fixable" for fixup_utoprim() */
    pflag[index(i,j,z)] = -333;
	pflag[index(N1_MPI_offset - N1G, N2_MPI_offset - N2G, N3_MPI_offset - N3G)] = 100;
    failimage[index(i,j,z)][3]++ ;
  }
  else { 
    if(gamma > GAMMAMAX) {
      f = sqrt(
	       (GAMMAMAX*GAMMAMAX - 1.)/
	       (gamma*gamma - 1.)
	       ) ;
      pv[U1] *= f ;	
      pv[U2] *= f ;	
      pv[U3] *= f ;	
    }
  }

  return;
}


/**************************************************************************************
 INTERPOLATION STENCILS:  
 ------------------------
   -- let the stencils be characterized by the following numbering convention:

           1 2 3 
           8 x 4      where x is the point at which we are interpolating 
           7 6 5
*******************************************************************************************/

/* 12345678 */
#define AVG8(pr,i,j,z,k)  \
        (0.125*(pr[index(i-1,j+1,z)][k]+pr[index(i,j+1,z)][k]+pr[index(i+1,j+1,z)][k]+pr[index(i+1,j,z)][k]+pr[index(i+1,j-1,z)][k]+pr[index(i,j-1,z)][k]+pr[index(i-1,j-1,z)][k]+pr[index(i-1,j,z)][k])) 

/* 2468  */
#define AVG4_1(pr,i,j,z,k) (0.25*(pr[index(i,j+1,z)][k]+pr[index(i,j-1,z)][k]+pr[index(i-1,j,z)][k]+pr[index(i+1,j,z)][k]))

/* 1357  */
#define AVG4_2(pr,i,j,z, k) (0.25*(pr[index(i+1,j+1,z)][k]+pr[index(i+1,j-1,z)][k]+pr[index(i-1,j+1,z)][k]+pr[index(i-1,j-1,z)][k]))

/* 2468+cells in 3rd dimension  */
#define AVG6_1(pr,i,j,z,k) (1./6.*(pr[index(i,j+1,z)][k]+pr[index(i,j-1,z)][k]+pr[index(i-1,j,z)][k]+pr[index(i+1,j,z)][k] +pr[index(i,j,z+1)][k]+pr[index(i,j,z-1)][k]))

/* 2468+cells in 3rd dimension  */
#define AVG6_2(pr,i,j,z,k) (1./6.*(pr[index(i+1,j+1,z)][k]+pr[index(i+1,j-1,z)][k]+pr[index(i-1,j+1,z)][k]+pr[index(i-1,j-1,z)][k] +pr[index(i,j,z+1)][k]+pr[index(i,j,z-1)][k]))

/* + shaped,  Linear interpolation in X1 or X2 directions using only neighbors in these direction */
/* 48  */
#define AVG2_X1(pr,i,j,z,k) (0.5*(pr[index(i-1,j,z)][k]+pr[index(i+1,j,z)][k]))
/* 26  */
#define AVG2_X2(pr,i,j,z,k) (0.5*(pr[index(i,j-1,z)][k]+pr[index(i,j+1,z)][k]))
/*910*/
#define AVG2_X3(pr,i,j,z,k) (0.5*(pr[index(i,j,z-1)][k]+pr[index(i,j,z+1)][k]))

/* x shaped,  Linear interpolation diagonally along both X1 and X2 directions "corner" neighbors */
/* 37  */
#define AVG2_1_X1X2(pr,i,j,z,k) (0.5*(pr[index(i-1,j-1,z)][k]+pr[index(i+1,j+1,z)][k]))
/* 15  */
#define AVG2_2_X1X2(pr,i,j,z,k) (0.5*(pr[index(i-1,j+1,z)][k]+pr[index(i+1,j-1,z)][k]))

/*******************************************************************************************
  fixup_utoprim(): 

    -- figures out (w/ pflag[]) which stencil to use to interpolate bad point from neighbors;

    -- here we use the following numbering scheme for the neighboring cells to i,j:  

                      1  2  3 
                      8  x  4        where "x" is the (i,j) cell or the cell to be interpolated
                      7  6  5

 *******************************************************************************************/

void fixup_utoprim( double ((*pv)[NPR]) )  
{
  int i, j, z, k;
  static int pf[11];

  /* Fix the interior points first */ 

#pragma omp parallel shared(pflag, pv) private(i,j,z,k, pf)
  {
	#pragma omp for schedule(static,1)
	  ZSLOOP3D(N1_MPI_offset, N1_MPI_offset + N1_MPI - 1, N2_MPI_offset, N2_MPI_offset + N2_MPI - 1, N3_MPI_offset, N3_MPI_offset + N3_MPI - 1) 	{
		  if (pflag[index(i,j,z)] != 0) {
			  //printf("i: %d j: %d, pflag: %d \n", i, j, pflag[i][j]);
			  pf[1] = !pflag[index(i - 1, j + 1, z)];   pf[2] = !pflag[index(i, j + 1, z)];  pf[3] = !pflag[index(i + 1, j + 1, z)];
			  pf[8] = !pflag[index(i - 1, j, z)];                           pf[4] = !pflag[index(i + 1, j, z)];
			  pf[7] = !pflag[index(i-1, j-1, z)];   pf[6] = !pflag[index(i, j - 1, z)];  pf[5] = !pflag[index(i + 1, j - 1, z)];
			  #if(N3>1)
			  pf[9] = !pflag[index(i, j, z + 1)]; pf[10] = !pflag[index(i, j, z - 1)];
			  #else
			  pf[9]=0;						      pf[10]=0;
			  #endif
			  /* Now the pf's  are true if they represent good points */

			  //      if(      pf[1]&&pf[2]&&pf[3]&&pf[4]&&pf[5]&&pf[6]&&pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = AVG8(            pv,i,j,k)                   ; }
			  //      else if(        pf[2]&&       pf[4]&&       pf[6]&&       pf[8] ){ FLOOP pv[i][j][k] = AVG4_1(          pv,i,j,k)                   ; }
			  //      else if( pf[1]&&       pf[3]&&       pf[5]&&       pf[7]        ){ FLOOP pv[i][j][k] = AVG4_2(          pv,i,j,k)                   ; }
			  //      else if(               pf[3]&&pf[4]&&              pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
			  //      else if(        pf[2]&&pf[3]&&              pf[6]&&pf[7]        ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
			  //      else if( pf[1]&&              pf[4]&&pf[5]&&              pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
			  //      else if( pf[1]&&pf[2]&&              pf[5]&&pf[6]               ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
			  //      else if(               pf[3]&&                     pf[7]        ){ FLOOP pv[i][j][k] = AVG2_1_X1X2(     pv,i,j,k)                   ; }
			  //      else if( pf[1]&&                     pf[5]                      ){ FLOOP pv[i][j][k] = AVG2_2_X1X2(     pv,i,j,k)                   ; }
			  //      else if(        pf[2]&&                     pf[6]               ){ FLOOP pv[i][j][k] = AVG2_X2(         pv,i,j,k)                   ; }
			  //      else if(                      pf[4]&&                     pf[8] ){ FLOOP pv[i][j][k] = AVG2_X1(         pv,i,j,k)                   ; }

			  // Old way:
			  if (pf[2] && pf[4] && pf[6] && pf[8] && pf[9] && pf[10]){
					#pragma simd
				  FLOOP pv[index(i,j,z)][k] = AVG6_1(pv, i, j, z, k);
			  }
			  else if (pf[1] && pf[3] && pf[5] && pf[7] && pf[9] && pf[10]){
				#pragma simd
				  FLOOP pv[index(i, j, z)][k] = AVG6_2(pv, i, j, z, k);
			  }
			  else if (pf[2] && pf[4] && pf[6] && pf[8]){
				#pragma simd
				  FLOOP pv[index(i,j,z)][k] = AVG4_1(pv, i, j, z, k);
			  }
			  else if (pf[1] && pf[3] && pf[5] && pf[7]){ 
				#pragma simd
					FLOOP pv[index(i,j,z)][k] = AVG4_2(pv, i, j,z, k); 
			  }
			  else if (pf[2] && pf[6]){
				#pragma simd
				  FLOOP pv[index(i, j, z)][k] = AVG2_X1(pv, i, j, z, k);
			  }
			  else if (pf[4] && pf[8]){
				#pragma simd
				  FLOOP pv[index(i, j, z)][k] = AVG2_X2(pv, i, j, z, k);
			  }
			  else if (pf[9] && pf[10]){
				#pragma simd
				  FLOOP pv[index(i, j, z)][k] = AVG2_X3(pv, i, j, z, k);
			  }
			  else{
				  failimage[index(i,j,z)][4]++;
				 
				  /* if nothing better to do, then leave densities and B-field unchanged, set v^i = 0 */
				#pragma simd
				  for (k = RHO; k <= UU; k++) { pv[index(i,j,z)][k] = 0.5*(AVG4_1(pv, i, j,z, k) + AVG4_2(pv, i, j, z, k)); }
				  pv[index(i,j,z)][U1] = pv[index(i,j,z)][U2] = pv[index(i,j,z)][U3] = 0.;
			  }
			  pflag[index(i,j,z)] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
			  //fixup1zone(i, j,z, pv[index(i,j,z)]);  /* Floor and limit gamma the interpolated value */
		  }
	  }
  }
  return;
}


#if( DO_FONT_FIX ) 
/***********************************************************************
   set_Katm():

       -- sets the EOS constant used for Font's fix. 

       -- see utoprim_1dfix1.c and utoprim_1dvsq2fix1.c  for more
           information. 

       -- uses the initial floor values of rho/u determined by fixup1zone()

       -- we assume here that Constant X1,r is independent of theta,X2

***********************************************************************/
void set_Katm( void )
{
  int i, j, k, G_type ;
  double prim[NPR], G_tmp;

  j = 1;

  G_type = get_G_ATM( &G_tmp );

  if (rank == 0){
	  fflush(stdout);
	  fprintf(stdout, "G_tmp = %26.20e \n", G_tmp);
	  fflush(stdout);
  }

  j = 0;
  for (i = N1_MPI_offset; i <N1_MPI_offset+ N1_MPI; i++) {
    PLOOP prim[k] = 0.;
    prim[RHO] = prim[UU] = -1.;

    fixup1zone( i, j, 0, prim );
    Katm[i] = (gam - 1.) * prim[UU] / pow( prim[RHO], G_tmp ) ;
    
    //fflush(stdout);
    //fprintf(stdout,"Katm[%d] = %26.20e \n", i, Katm[i] );
	//fflush(stdout); 
  }

  return;
}
#endif


#undef FLOOP 
