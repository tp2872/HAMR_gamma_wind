
#include "decs.h"

#if(DOPARTICLES)

#define EPS 1.e-6
void advance_particles(double(*restrict pr[NB_LOCAL])[NPR], double Dt, int flag)
{
	int z, l, i, j, m, k;
	double ucon[NDIM], vel[NDIM], beta[NDIM], gamma[NDIM][NDIM];
    double X[NDIM], Xh[NDIM], Xl[NDIM];
    double xcon_tmp[NDIM], pcov_tmp[NDIM];
    double c1[NDIM], c2[NDIM], d1[NDIM], d2[NDIM];
    double gcona[NDIM][NDIM], gconh[NDIM][NDIM], gcovh[NDIM][NDIM];
    double gcova[NDIM][NDIM], gconl[NDIM][NDIM], gcovl[NDIM][NDIM];
    double A[NPTOT][NDIM][NDIM], B[NPTOT][NDIM][NDIM], C[NPTOT][NDIM][NDIM];
	double f1, f2, f3, alpha;
	struct of_geom geom;
    int n = 0;
    double p_t, put_2;

#if 0
	for (l = 0; l < NPTOT; l++) {
        
    //ensure that phi-periodicity is in place
    if (xcon_p[l][3] >= startx[3] + (BS_3) * dx[nl[n]][3]) {
        xcon_p[l][3] -= (BS_3) * dx[nl[n]][3];
    }
    else if (xcon_p[l][3] < startx[3]) {
        xcon_p[l][3] += (BS_3) * dx[nl[n]][3];
    }

    /* don't update particles that are off-grid for this MPI process */
        if(xcon_p[l][1] >= startx[1] && xcon_p[l][2] >= startx[2] && xcon_p[l][3] >= startx[3] && xcon_p[l][1] < startx[1] + (BS_1) * dx[nl[n]][1] && xcon_p[l][2] < startx[2] + (BS_2) * dx[nl[n]][2] && xcon_p[l][3] < startx[3] + (BS_3) * dx[nl[n]][3]) {
            
            /* the four-velocities are zone-centered */
            f1 = (xcon_p[l][1] - startx[1] + 0.5*dx[nl[n]][1]) / dx[nl[n]][1];
            f2 = (xcon_p[l][2] - startx[2] + 0.5*dx[nl[n]][2]) / dx[nl[n]][2];
            f3 = (xcon_p[l][3] - startx[3] + 0.5*dx[nl[n]][3]) / dx[nl[n]][3];

            /* find nearest zone center */
            i = lround( f1 ) ;
            j = lround( f2 ) ;
            z = lround( f3 ) ;

            get_geometry(n_ord[n], i, j, z, CENT, &geom);
            ucon_calc(pr[nl[n]][index_3D(n, i, j, z)], &geom, ucon);
            for (m = 1; m < NDIM; m++) vel[m] = ucon[m]/ucon[0];

            /* push particle forward */
            for (m = 1; m < NDIM; m++) xcon_p[l][m] += Dt * vel[m];

            }
	}
#endif
    
    for (m = 0; m < NPTOT; m++) {
        //ensure that phi-periodicity is in place
        if (xcon_p[m][3] >= startx[3] + (BS_3) * dx[nl[n]][3]) {
            xcon_p[m][3] -= (BS_3) * dx[nl[n]][3];
        }
        else if (xcon_p[m][3] < startx[3]) {
            xcon_p[m][3] += (BS_3) * dx[nl[n]][3];
        }
        
        /* don't update particles that are off-grid for this MPI process */
        if(xcon_p[m][1] >= startx[1] && xcon_p[m][2] >= startx[2] && xcon_p[m][3] >= startx[3] && xcon_p[m][1] < startx[1] + (BS_1) * dx[nl[n]][1] && xcon_p[m][2] < startx[2] + (BS_2) * dx[nl[n]][2] && xcon_p[m][3] < startx[3] + (BS_3) * dx[nl[n]][3]) {
            
            gcov_func(xcon_p[m],gcova);
            gcon_func(gcova,gcona);
            
            for (j=0; j<NDIM; j++) {
                xcon_tmp[j] = 0.0;
                pcov_tmp[j] = 0.0;
            }
            
            // Calculate c1
            c1[0] = 1.0;
            for (i=1; i<NDIM; i++) {
                c1[i] = gcona[0][i] / gcona[0][0];
                for (j=1; j<NDIM; j++) c1[i] += (gcona[i][j] - gcona[0][i] * gcona[0][j] / gcona[0][0]) * pcov_p[m][j] / pcov_p[m][0];
            }
        
            for (j=1; j<NDIM; j++) xcon_tmp[j] = xcon_p[m][j] + Dt * c1[j];
            
            // Calculate d1
            d1[0] = 0.0;
            
            p_t = pcov_p[m][0];
            for (j=1; j<NDIM; j++) p_t -= gcona[0][j] * pcov_p[m][j];
            p_t *= 1.0 / gcona[0][0];
            
            for(i=1;i<NDIM;i++) {
                for(l=0;l<NDIM;l++) Xh[l] = xcon_p[m][l];
                for(l=0;l<NDIM;l++) Xl[l] = xcon_p[m][l];
                
                Xh[i] += EPS;
                Xl[i] -= EPS;
                
                gcov_func(Xh,gcovh);
                gcov_func(Xl,gcovl);
                gcon_func(gcovh,gconh);
                gcon_func(gcovl,gconl);
                
                d1[i] = (gconh[0][0] - gconl[0][0]) / (Xh[i] - Xl[i]) * p_t * p_t;
                for (j=1; j<NDIM; j++) d1[i] += 2.0 * (gconh[0][j] - gconl[0][j]) / (Xh[i] - Xl[i]) * p_t * pcov_p[m][j];
                for (j=1; j<NDIM; j++) for (k=1; k<NDIM; k++) d1[i] += (gconh[j][k] - gconl[j][k]) / (Xh[i] - Xl[i]) * pcov_p[m][j] * pcov_p[m][k];
                d1[i] *= -0.5 / pcov_p[m][0];
            }
            
            for (j=1; j<NDIM; j++) pcov_tmp[j] = pcov_p[m][j] + Dt * d1[j];
            
            gcov_func(xcon_tmp,gcova);
            gcon_func(gcova,gcona);
            
            put_2 = 0.0;
            for (j=1; j<NDIM; j++) for (k=1; k<NDIM; k++) put_2 += (-gcona[0][0] * gcona[j][k] + gcona[0][j] * gcona[0][k]) * pcov_tmp[j] * pcov_tmp[k];
            if (put_2 > 0.0) pcov_tmp[0] = sqrt(put_2);
            else fprintf(stderr, "Error: particle P^t complex.\n");
            
            // Calculate c2
            c2[0] = 1.0;
            for (i=1; i<NDIM; i++) {
                c2[i] = gcona[0][i] / gcona[0][0];
                for (j=1; j<NDIM; j++) c2[i] += (gcona[i][j] - gcona[0][i] * gcona[0][j] / gcona[0][0]) * pcov_tmp[j] / pcov_tmp[0];
            }

            for (j=1; j<NDIM; j++) xcon_p[m][j] = xcon_p[m][j] + Dt * 0.5*(c1[j] + c2[j]);
            
            // Calculate d2
            d2[0] = 0.0;
            
            p_t = pcov_tmp[0];
            for (j=1; j<NDIM; j++) p_t -= gcona[0][j] * pcov_tmp[j];
            p_t *= 1.0 / gcona[0][0];
            
            for(i=1;i<NDIM;i++) {
                for(l=0;l<NDIM;l++) Xh[l] = xcon_tmp[l];
                for(l=0;l<NDIM;l++) Xl[l] = xcon_tmp[l];
                
                Xh[i] += EPS;
                Xl[i] -= EPS;
                
                gcov_func(Xh,gcovh);
                gcov_func(Xl,gcovl);
                gcon_func(gcovh,gconh);
                gcon_func(gcovl,gconl);
                
                d2[i] = (gconh[0][0] - gconl[0][0]) / (Xh[i] - Xl[i]) * p_t * p_t;
                for (j=1; j<NDIM; j++) d2[i] += 2.0 * (gconh[0][j] - gconl[0][j]) / (Xh[i] - Xl[i]) * p_t * pcov_tmp[j];
                for (j=1; j<NDIM; j++) for (k=1; k<NDIM; k++) d2[i] += (gconh[j][k] - gconl[j][k]) / (Xh[i] - Xl[i]) * pcov_tmp[j] * pcov_tmp[k];
                d2[i] *= -0.5 / pcov_tmp[0];
            }
            
            for (j=1; j<NDIM; j++) pcov_p[m][j] = pcov_p[m][j] + Dt * 0.5*(d1[j] + d2[j]);
            
            gcov_func(xcon_p[m],gcova);
            gcon_func(gcova,gcona);
            put_2 = 0.0;
            for (j=1; j<NDIM; j++) for (k=1; k<NDIM; k++) put_2 += (-gcona[0][0] * gcona[j][k] + gcona[0][j] * gcona[0][k]) * pcov_p[m][j] * pcov_p[m][k];
            if (put_2 > 0.0) pcov_p[m][0] = sqrt(put_2);
            else fprintf(stderr, "Error: particle P^t complex.\n");
            
            // Output p_t to see energy conservation
            p_t = pcov_p[m][0];
            for (j=1; j<NDIM; j++) p_t -= gcona[0][j] * pcov_p[m][j];
            p_t *= 1.0 / gcona[0][0];
            
            xcon_p[m][0] = p_t;
            
            //fprintf(stderr, "m = %d, p^t = %f, p_t = %e, p_t_ini = %e, p_t_err = %e\n ## ", m, pcov_p[m][0], p_t, xcon_p[m][0], (p_t - xcon_p[m][0]));
        }
    }
}
#undef EPS


void init_particles()
{
    int ii, jj, zz;
    int i, j, z, k, l, m;
    double X[NDIM];
    double gcova[NDIM][NDIM], gcona[NDIM][NDIM];
    double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
    double p_prime[NDIM];
    struct of_geom geom;
    double rancval1, rancval2, rancval3, alpha, nueps, angle, anglep;

    int n = 0;

    for (m = 0; m < NPTOT; m++) {
        //i = (int) (ranc(0) * ((log(2 * rmax) - startx[1]) / dx[nl[n]][1] - ((log(6.0) - startx[1]) / dx[nl[n]][1]))) + (int) ((log(6.0) - startx[1]) / dx[nl[n]][1]) + 1; //(ranc(0) * BS_1 / 2) + BS_1/4;
        ii = (int) (ranc(0) * BS_1); // 21 = 2 * r_isco
        jj = BS_2/2; //(int) (ranc(0) * BS_2); // BS_2/2; //(int) (ranc(0) * BS_2);
        zz = (int) (ranc(0) * BS_3);
        
        coord(n_ord[n], ii, jj, zz, CORN, X);

        rancval1 = ranc(0);
        //rancval2 = ranc(0);
        rancval3 = ranc(0);

        //xcon_p[m][0] = (double) k; //particle number is its tag
        xcon_p[m][1] = log(10.0); //X[1]; // + rancval1 * dx[nl[n]][1];
        xcon_p[m][2] = X[2]; // + rancval2 * dx[nl[n]][2];
        xcon_p[m][3] = 0.0; //X[3]; // + rancval3 * dx[nl[n]][3];
        
        gcov_func(xcon_p[m],gcova);
        gcon_func(gcova,gcona);
        
        build_tetrad(gcona, Econ, Ecov);
        
        nueps = 5.0;
        angle = M_PI_2; //sranc(0) * 2.0 * M_PI;
        anglep = ranc(0) * 2.0 * M_PI;
        
        p_prime[0] = 0.0;
        p_prime[1] = (cos(anglep));
        p_prime[2] = (sin(anglep) * cos(angle));
        p_prime[3] = (sin(anglep) * sin(angle));
        
        for (i=0; i<NDIM; i++) {
            pcov_p[m][i] = nueps * Ecov[0][i];
            for (j=1; j<NDIM; j++) pcov_p[m][i] += nueps * Ecov[j][i] * p_prime[j];
        }
        
        // Check that p_mu is a null-vector
        double put = nueps * Econ[0][0];
        for (i=1; i<NDIM; i++) put += nueps * Econ[i][0] * p_prime[i];
        
        double check_null = 0.0;
        for (i=0; i<NDIM; i++) for (j=0; j<NDIM; j++) check_null += gcona[i][j] * pcov_p[m][i] * pcov_p[m][j];
        
        xcon_p[m][0] = pcov_p[m][0];
        pcov_p[m][0] = put;
        
        fprintf(stderr, "Particle no. %d, Check = %e\n", m, check_null);
    }
}

void build_tetrad(double gcon[NDIM][NDIM], double Econ[NDIM][NDIM], double Ecov[NDIM][NDIM]) {
    int i, j, k;
    double alpha;
    
    alpha = 1.0/sqrt(-gcon[0][0]);
    
    // Time-like basis: n^mu
    Ecov[0][0] = -alpha;
    for (j=1; j<NDIM; j++) Ecov[0][j] = 0.0;
    for (j=0; j<NDIM; j++) Econ[0][j] = 0.0;
    for (j=0; j<NDIM; j++) for (k=0; k<NDIM; k++) Econ[0][j] += gcon[j][k] * Ecov[0][k];
    
    // Other basis vectors : Gram-Schmidt algorithm on coordinate basis vectors
    for (i=1; i<NDIM; i++) {
        
        // Initiate vectors
        for (j=0; j<NDIM; j++) Ecov[i][j] = (i==j ? 1.0 : 0.0);
        
        // Make this vector orthogonal to the existing components of tetrad
        for (j=0; j<i; j++) {
            double dotproduct = 0.0;
            double sign_j = (j==0 ? -1.0 : 1.0);
            for (k=0; k<NDIM; k++) dotproduct += Econ[j][k] * Ecov[i][k];
            for (k=0; k<NDIM; k++) Ecov[i][k] -= sign_j * dotproduct * Ecov[j][k];
        }
        
        for (j=0; j<NDIM; j++) {
            Econ[i][j] = 0.0;
            for (k=0; k<NDIM; k++) Econ[i][j] += gcon[j][k] * Ecov[i][k];
        }
        
        // Normalize
        double norm = 0.0;
        for (j=0; j<NDIM; j++) norm += Econ[i][j] * Ecov[i][j];
        norm = 1.0 / sqrt(norm);
        for (j=0; j<NDIM; j++) {
            Ecov[i][j] *= norm;
            Econ[i][j] *= norm;
        }
    }
}
#endif
