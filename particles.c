
#include "decs.h"

/*

 advance particle positions using fluid half-step primitives 

 revised cfg 11 apr 2016 to improve interpolation scheme 

*/
#if(DOPARTICLES)

#if 0
void advance_particles(double ph[N1M][N2M][N3M][NPR], double Dt)
{
	int z, l, i, j;
	double ucon[NDIM], vel[NDIM];
	double f1,f2,f3;
	struct of_geom geom;

	for (l = 0; l < myNp; l++) {
#if(BL)
                //ensure that phi-periodicity is in place
                //SASMARK: only works for phi-periodic grids, so do this only for BL!=0
                if (x_p[l][3] >= startx[3] + lenx[3]) {
                    x_p[l][3] -= lenx[3];
                }
                else if (x_p[l][3] < startx[3]) {
                    x_p[l][3] += lenx[3];
                }
#endif
		/* don't update particles that are off-grid for this MPI process */
		if(x_p[l][1] >= mpi_startx[1] && x_p[l][2] >= mpi_startx[2] && x_p[l][3] >= mpi_startx[3] &&
		   x_p[l][1]  < mpi_stopx[1]  && x_p[l][2]  < mpi_stopx[2]  && x_p[l][3]  < mpi_stopx[3] ) {

			/* the four-velocities are zone-centered */
			f1 = (x_p[l][1] - mpi_startx[1] + 0.5*dx[1]) / dx[1];
			f2 = (x_p[l][2] - mpi_startx[2] + 0.5*dx[2]) / dx[2];
            f3 = (x_p[l][3] - mpi_startx[3] + 0.5*dx[3]) / dx[3];

		   	/* find nearest zone center */
			i = lround( f1 ) ;
			j = lround( f2 ) ;
            z = lround( f3 ) ;

                        get_geometry(i, j, z, CENT, &geom);
                        ucon_calc(ph[i][j][z], &geom, ucon);
                        for (z = 1; z < NDIM; z++) vel[z] = ucon[z]/ucon[0];

			/* push particle forward */
			for (z = 1; z < NDIM; z++)
				x_p[l][z] += Dt * vel[z];

		}

	}
	/* done! */
}
#endif

/*

 initialize Lagrangian tracer particles
 cfg 4 feb 09

 simplified 10 apr 2016 cfg

*/

void init_particles()
{
    int i, j, z, k;
    double X[NDIM], x_p[NPTOT][NDIM];
    struct of_geom geom;
    double rancval1, rancval2, rancval3;

// DANAT: edits Jul 6 - start
    int n = 0;

    for (k = 0; k < NPTOT; k++) {
        i = (int) (ranc(0) * BS_1);
        j = (int) (ranc(0) * BS_2);
        z = (int) (ranc(0) * BS_3);
        
        get_geometry(n_ord[n], i, j, z, CENT, &geom);
        coord(n_ord[n], i, j, z, CORN, X);

        rancval1 = ranc(0);
        rancval2 = ranc(0);
        rancval3 = ranc(0);

        x_p[k][0] = (double) k; //particle number is its tag
        x_p[k][1] = X[1] + rancval1 * dx[nl[n_ord[n]]][1];
        x_p[k][2] = X[2] + rancval2 * dx[nl[n_ord[n]]][2];
        x_p[k][3] = X[3] + rancval3 * dx[nl[n_ord[n]]][3];
        fprintf(stderr, "MC particles: k = %d, [%d %d %d], (%f %f %f) \n", k, i, j, z, x_p[k][1], x_p[k][2], x_p[k][3]);
    }
    exit(1);
// DANAT: edits Jul 6 - end

}
#endif
