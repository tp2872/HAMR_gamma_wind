
#include "decs.h"

/*

 advance particle positions using fluid half-step primitives 

 revised cfg 11 apr 2016 to improve interpolation scheme 

*/
#if(DOPARTICLES)

#undef EPS
#define EPS 1.e-5

void advance_particles(double(*restrict pr[NB_LOCAL])[NPR], double Dt)
{
	int z, l, i, j, m, k;
	double ucon[NDIM], vel[NDIM], beta[NDIM], gamma[NDIM][NDIM];
    double X[NDIM], Xh[NDIM], Xl[NDIM];
    double gcona[NDIM][NDIM], gconh[NDIM][NDIM], gcovh[NDIM][NDIM];
    double gcova[NDIM][NDIM], gconl[NDIM][NDIM], gcovl[NDIM][NDIM];
    double A[NPTOT][NDIM][NDIM], B[NPTOT][NDIM][NDIM], C[NPTOT][NDIM][NDIM];
	double f1, f2, f3, alpha;
	struct of_geom geom;
    int n = 0;

#if 0
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
#endif
    
    for (m = 0; m < NPTOT; m++) {
        //ensure that phi-periodicity is in place
        if (x_p[m][3] >= startx[3] + (BS_3) * dx[nl[n]][3]) {
            x_p[m][3] -= (BS_3) * dx[nl[n]][3];
        }
        else if (x_p[m][3] < startx[3]) {
            x_p[m][3] += (BS_3) * dx[nl[n]][3];
        }
        
        /* don't update particles that are off-grid for this MPI process */
        if(x_p[m][1] >= startx[1] && x_p[m][2] >= startx[2] && x_p[m][3] >= startx[3] && x_p[m][1] < startx[1] + (BS_1) * dx[nl[n]][1] && x_p[m][2] < startx[2] + (BS_2) * dx[nl[n]][2] && x_p[m][3] < startx[3] + (BS_3) * dx[nl[n]][3]) {
            
            for(l=0;l<NDIM;l++) Xh[l] = x_p[m][l];
            gcov_func(Xh,gcova);
            gcon_func(gcova,gcona);
            
            for(i=1;i<NDIM;i++) {
                for(l=0;l<NDIM;l++) Xh[l] = x_p[m][l];
                for(l=0;l<NDIM;l++) Xl[l] = x_p[m][l];
                
                Xh[i] += EPS;
                Xl[i] -= EPS;
                
                gcov_func(Xh,gcovh);
                gcov_func(Xl,gcovl);
                gcon_func(gcovh,gconh);
                gcon_func(gcovl,gconl);
                
                // Update p_i
                x_p[m][i+NDIM] += Dt * 0.5*(1./gconh[0][0] - 1./gconl[0][0]) * x_p[m][NDIM] / (Xh[i] - Xl[i]);
                SLOOPA x_p[m][i+NDIM] += - Dt * (gconh[0][j]/gconh[0][0] - gconl[0][j]/gconl[0][0]) * x_p[m][j+NDIM] / (Xh[i] - Xl[i]);
                SLOOP  x_p[m][i+NDIM] += - Dt * 0.5*(gconh[j][k] - gconh[0][j] * gconh[0][k] / gconh[0][0] - gconl[j][k] + gconl[0][j] * gconl[0][k] / gconl[0][0]) / (Xh[i] - Xl[i]) * x_p[m][j+NDIM] * x_p[m][k+NDIM] / x_p[m][NDIM];
                
                // Update x^i
                x_p[m][i] += Dt * gcona[0][i] / gcona[0][0];
                SLOOPA x_p[m][i] += Dt * (gcona[i][j] - gcona[0][i] * gcona[0][j] / gcona[0][0]) * x_p[m][j+NDIM] / x_p[m][NDIM];
            }
        }
        //fprintf(stderr, "MC particles: k = %d, (%f %f %f), p^t=%f, p_i=(%f %f %f)\n", m, x_p[m][1], x_p[m][2], x_p[m][3], x_p[m][4], x_p[m][5], x_p[m][6], x_p[m][7]);
    }
    //fprintf(stderr, "\n");
	/* done! */
}

/* 

 initialize Lagrangian tracer particles
 cfg 4 feb 09

 simplified 10 apr 2016 cfg

*/

void init_particles()
{
    int i, j, z, k, l;
    double X[NDIM], X1[NDIM];
    double gcova[NDIM][NDIM], gcona[NDIM][NDIM];
    struct of_geom geom;
    double rancval1, rancval2, rancval3, alpha, nueps, angle, pu1, pu2, pu3;
    //double r, th, phi;

// DANAT: edits Jul 6 - start
    int n = 0;

    for (k = 0; k < NPTOT; k++) {
        //i = (int) (ranc(0) * ((log(2 * rmax) - startx[1]) / dx[nl[n]][1] - ((log(6.0) - startx[1]) / dx[nl[n]][1]))) + (int) ((log(6.0) - startx[1]) / dx[nl[n]][1]) + 1; //(ranc(0) * BS_1 / 2) + BS_1/4;
        i = (int) (ranc(0) * 21); // 21 = 2 * r_isco
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
        
        for(l=0;l<NDIM;l++) X1[l] = x_p[k][l];
        gcov_func(X1,gcova);
        gcon_func(gcova,gcona);
        alpha = 1.0/sqrt(-gcona[0][0]);
        nueps = -1.0;
        angle = ranc(0) * 2. * M_PI;
        
        pu1 = nueps * (gcona[0][1]/gcona[0][0] + cos(angle));
        pu2 = 0.0;
        pu3 = nueps * (gcona[0][3]/gcona[0][0] + sin(angle));
        x_p[k][4] = nueps/alpha;
        x_p[k][5] = gcova[1][0] * nueps/alpha + gcova[1][1] * pu1 + gcova[1][2] * pu2 + gcova[1][3] * pu3;
        x_p[k][6] = gcova[2][0] * nueps/alpha + gcova[2][1] * pu1 + gcova[2][2] * pu2 + gcova[2][3] * pu3;
        x_p[k][7] = gcova[3][0] * nueps/alpha + gcova[3][1] * pu1 + gcova[3][2] * pu2 + gcova[3][3] * pu3;
        
        fprintf(stderr, "MC particles: k = %d, [%d %d %d], (%f %f %f), p^t=%f, p_i=(%f %f %f)\n", k, i, j, z, x_p[k][1], x_p[k][2], x_p[k][3], x_p[k][4], x_p[k][5], x_p[k][6], x_p[k][7]);
    }
    
// DANAT: edits Jul 6 - end

}
#endif
