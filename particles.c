
#include "decs.h"

/*

 advance particle positions using fluid half-step primitives 

 revised cfg 11 apr 2016 to improve interpolation scheme 

*/
#if(DOPARTICLES)

#if 1
void advance_particles(double(*restrict pr[NB_LOCAL])[NPR], double Dt)
{
	int z, l, i, j, m;
	double ucon[NDIM], vel[NDIM];
	double f1,f2,f3;
	struct of_geom geom;
    int n = 0;

	for (l = 0; l < NPTOT; l++) {
        
    //ensure that phi-periodicity is in place
    if (x_p[l][3] >= startx[3] + (BS_3) * dx[nl[n]][3]) {
        x_p[l][3] -= (BS_3) * dx[nl[n]][3];
    }
    else if (x_p[l][3] < startx[3]) {
        x_p[l][3] += (BS_3) * dx[nl[n]][3];
    }

    /* don't update particles that are off-grid for this MPI process */
        if(x_p[l][1] >= startx[1] && x_p[l][2] >= startx[2] && x_p[l][3] >= startx[3] && x_p[l][1] < startx[1] + (BS_1) * dx[nl[n]][1] && x_p[l][2] < startx[2] + (BS_2) * dx[nl[n]][2] && x_p[l][3] < startx[3] + (BS_3) * dx[nl[n]][3]) {
            
            /* the four-velocities are zone-centered */
            f1 = (x_p[l][1] - startx[1] + 0.5*dx[nl[n]][1]) / dx[nl[n]][1];
            f2 = (x_p[l][2] - startx[2] + 0.5*dx[nl[n]][2]) / dx[nl[n]][2];
            f3 = (x_p[l][3] - startx[3] + 0.5*dx[nl[n]][3]) / dx[nl[n]][3];

            /* find nearest zone center */
            i = lround( f1 ) ;
            j = lround( f2 ) ;
            z = lround( f3 ) ;

            get_geometry(n_ord[n], i, j, z, CENT, &geom);
            ucon_calc(pr[nl[n]][index_3D(n, i, j, z)], &geom, ucon);
            for (m = 1; m < NDIM; m++) vel[m] = ucon[m]/ucon[0];

            /* push particle forward */
            for (m = 1; m < NDIM; m++) x_p[l][m] += Dt * vel[m];

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
    double X[NDIM];
    struct of_geom geom;
    double rancval1, rancval2, rancval3;
    //double r, th, phi;

// DANAT: edits Jul 6 - start
    int n = 0;

    for (k = 0; k < NPTOT; k++) {
        i = (int) (ranc(0) * ((log(2 * rmax) - startx[1]) / dx[nl[n]][1] - ((log(6.0) - startx[1]) / dx[nl[n]][1]))) + (int) ((log(6.0) - startx[1]) / dx[nl[n]][1]) + 1; //(ranc(0) * BS_1 / 2) + BS_1/4;
        j = BS_2/2; //(int) (ranc(0) * BS_2);
        z = (int) (ranc(0) * BS_3);
        
        coord(n_ord[n], i, j, z, CORN, X);

        rancval1 = ranc(0);
        rancval2 = ranc(0);
        rancval3 = ranc(0);

        x_p[k][0] = (double) k; //particle number is its tag
        x_p[k][1] = X[1] + rancval1 * dx[nl[n]][1];
        x_p[k][2] = X[2]; // + rancval2 * dx[nl[n]] [2];
        x_p[k][3] = X[3] + rancval3 * dx[nl[n]][3];
        
        //bl_coord(x_p[k], &r, &th, &phi);
        
        //fprintf(stderr, "MC particles: k = %d, [%d %d %d], (%f %f %f), BL=(%f %f %f)\n", k, i, j, z, x_p[k][1], x_p[k][2], x_p[k][3], r, th, phi);
    }
// DANAT: edits Jul 6 - end

}
#endif
