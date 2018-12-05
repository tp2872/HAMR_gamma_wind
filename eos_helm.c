#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "Flash.h"
#include "Eos.h"
#include "Eos_map.h"
#define MAXLEN (1024)
#ifdef DEBUG_ALL
#define DEBUG_EOS
#endif

#define min(x,y) ({ \
	typeof(x) _x = (x);	\
	typeof(y) _y = (y);	\
	(void) (&_x == &_y);	\
	_x < _y ? _x : _y; })

#define max(x,y) ({ \
	typeof(x) _x = (x);	\
	typeof(y) _y = (y);	\
	(void) (&_x == &_y);	\
	_x > _y ? _x : _y; })

#define EOSIMAX (211)
#define EOSJMAX (71)
#define eos_tlo (4.0)
#define eos_dlo (-10.0)

// ***********Beginning of statement function declarations **********
// quintic hermite polynomial statement functions
// psi0 and its derivatives
double psi0(double zFunc){
    return zFunc*zFunc*zFunc * ( zFunc * (-6.0e0*zFunc + 15.0e0) -10.0e0) + 1.0e0;
}
double dpsi0(double zFunc){
    return zFunc*zFunc * ( zFunc * (-30.0e0*zFunc + 60.0e0) - 30.0e0);
}
double ddpsi0(double zFunc){
    return zFunc* ( zFunc*( -120.0e0*zFunc + 180.0e0) -60.0e0);
}

// psi1 and its derivatives
double psi1(double zFunc){
    return zFunc* ( zFunc*zFunc * ( zFunc * (-3.0e0*zFunc + 8.0e0) - 6.0e0) + 1.0e0);
}
double dpsi1(double zFunc){
    return zFunc*zFunc * ( zFunc * (-15.0e0*zFunc + 32.0e0) - 18.0e0) +1.0e0;
}
double ddpsi1(double zFunc){
    return zFunc * (zFunc * (-60.0e0*zFunc + 96.0e0) -36.0e0);
}

// psi2  and its derivatives
double psi2(double zFunc){
    return 0.5e0*zFunc*zFunc*( zFunc* ( zFunc * (-zFunc + 3.0e0) - 3.0e0) + 1.0e0);
}
double dpsi2(double zFunc){
    return 0.5e0*zFunc*( zFunc*(zFunc*(-5.0e0*zFunc + 12.0e0) - 9.0e0) + 2.0e0);
}
double ddpsi2(double zFunc){
    return 0.5e0*(zFunc*( zFunc * (-20.0e0*zFunc + 36.0e0) - 18.0e0) + 2.0e0);
}


double h5(double w0t, double w1t, double w2t, double w0mt, double w1mt, double w2mt,
          double w0d, double w1d, double w2d, double w0md, double w1md, double w2md,
          double *fi){
    return fi[0]  *w0d*w0t   + fi[1]  *w0md*w0t \
            + fi[2]  *w0d*w0mt  + fi[3]  *w0md*w0mt \
            + fi[4]  *w0d*w1t   + fi[5]  *w0md*w1t \
            + fi[6]  *w0d*w1mt  + fi[7]  *w0md*w1mt \
            + fi[8]  *w0d*w2t   + fi[9] *w0md*w2t \
            + fi[10] *w0d*w2mt  + fi[11] *w0md*w2mt \
            + fi[12] *w1d*w0t   + fi[13] *w1md*w0t \
            + fi[14] *w1d*w0mt  + fi[15] *w1md*w0mt \
            + fi[16] *w2d*w0t   + fi[17] *w2md*w0t \
            + fi[18] *w2d*w0mt  + fi[19] *w2md*w0mt \
            + fi[20] *w1d*w1t   + fi[21] *w1md*w1t \
            + fi[22] *w1d*w1mt  + fi[23] *w1md*w1mt \
            + fi[24] *w2d*w1t   + fi[25] *w2md*w1t \
            + fi[26] *w2d*w1mt  + fi[27] *w2md*w1mt \
            + fi[28] *w1d*w2t   + fi[29] *w1md*w2t \
            + fi[30] *w1d*w2mt  + fi[31] *w1md*w2mt \
            + fi[32] *w2d*w2t   + fi[33] *w2md*w2t \
            + fi[34] *w2d*w2mt  + fi[35] *w2md*w2mt;
}

//  cubic hermite polynomial statement functions
//  psi0 & derivatives
double xpsi0(double zFunc){
    return zFunc * zFunc * (2.0e0*zFunc - 3.0e0) + 1.0;
}
//  psi1 & derivatives
double xpsi1(double zFunc){
    return zFunc * ( zFunc * (zFunc - 2.0e0) + 1.0e0);
}
//************ This is the end statement function definitions *************

void eos_init (
               double eos_f[EOSIMAX][EOSJMAX], double eos_fd[EOSIMAX][EOSJMAX], double eos_ft[EOSIMAX][EOSJMAX], double eos_fdd[EOSIMAX][EOSJMAX], double eos_ftt[EOSIMAX][EOSJMAX],
               double eos_fdt[EOSIMAX][EOSJMAX], double eos_fddt[EOSIMAX][EOSJMAX], double eos_fdtt[EOSIMAX][EOSJMAX], double eos_fddtt[EOSIMAX][EOSJMAX],
               double eos_dpdf[EOSIMAX][EOSJMAX], double eos_dpdfd[EOSIMAX][EOSJMAX], double eos_dpdft[EOSIMAX][EOSJMAX], double eos_dpdfdt[EOSIMAX][EOSJMAX],
               double eos_ef[EOSIMAX][EOSJMAX], double eos_efd[EOSIMAX][EOSJMAX], double eos_eft[EOSIMAX][EOSJMAX], double eos_efdt[EOSIMAX][EOSJMAX],
               double eos_xf[EOSIMAX][EOSJMAX], double eos_xfd[EOSIMAX][EOSJMAX], double eos_xft[EOSIMAX][EOSJMAX], double eos_xfdt[EOSIMAX][EOSJMAX],
               double eos_t[EOSJMAX], double eos_d[EOSIMAX], double eos_dd[EOSIMAX], double eos_ddSqr[EOSIMAX], double eos_ddInv[EOSIMAX],
               double eos_ddSqrInv[EOSIMAX], double eos_dt[EOSJMAX], double eos_dtSqr[EOSJMAX], double eos_dtInv[EOSJMAX], double eos_dtSqrInv[EOSJMAX]
               ) {
    
    FILE *fp;
    char fname_eos_table[] = "helm_table.dat";
    char buf[MAXLEN], *ptr;
    int nitems_read, nitems_expected;
    
    int i, j;
    double tstp, dstp;
    double eos_tstpi, eos_dstpi;
    
    fp = fopen(fname_eos_table, "r");
    if (NULL == fp) {
        fprintf(stderr, "Couldn't open %s for reading, exiting\n", fname_eos_table);
        exit(1234);
    }
    
    //..read the helmholtz free energy table
    for (j=0; j<EOSJMAX; j++)  {
        for (i=0; i<EOSIMAX; i++) {
            ptr = fgets(buf, MAXLEN, fp);
            if (NULL == ptr) break;
            nitems_read = sscanf(ptr,  "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                                 &eos_f[i][j], &eos_fd[i][j], &eos_ft[i][j], &eos_fdd[i][j],
                                 &eos_ftt[i][j], &eos_fdt[i][j], &eos_fddt[i][j], &eos_fdtt[i][j], &eos_fddtt[i][j]);
            nitems_expected = 9;
            if (nitems_expected != nitems_read) break;
        }
    }
    
    //..read the pressure derivative with density table
    for (j=0; j<EOSJMAX; j++)  {
        for (i=0; i<EOSIMAX; i++) {
            ptr = fgets(buf, MAXLEN, fp);
            if (NULL == ptr) break;
            nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n",
                                 &eos_dpdf[i][j], &eos_dpdfd[i][j], &eos_dpdft[i][j], &eos_dpdfdt[i][j]);
            nitems_expected = 4;
            if (nitems_expected != nitems_read) break;
        }
    }
    
    //..read the electron chemical potential table
    for (j=0; j<EOSJMAX; j++)  {
        for (i=0; i<EOSIMAX; i++) {
            ptr = fgets(buf, MAXLEN, fp);
            if (NULL == ptr) break;
            nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n",
                                 &eos_ef[i][j], &eos_efd[i][j], &eos_eft[i][j], &eos_efdt[i][j]);
            nitems_expected = 4;
            if (nitems_expected != nitems_read) break;
        }
    }
    
    //..read the number density table
    for (j=0; j<EOSJMAX; j++)  {
        for (i=0; i<EOSIMAX; i++) {
            ptr = fgets(buf, MAXLEN, fp);
            if (NULL == ptr) break;
            nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n",
                                 &eos_xf[i][j], &eos_xfd[i][j], &eos_xft[i][j], &eos_xfdt[i][j]);
            nitems_expected = 4;
            if (nitems_expected != nitems_read) break;
        }
    }
    
    fclose(fp);
    
    tstp  = (11.0e0 - eos_tlo)/(double)(EOSJMAX-1);
    eos_tstpi = 1.0e0/tstp;
    dstp  = (11.0e0 - eos_dlo)/(double)(EOSIMAX-1);
    eos_dstpi = 1.0e0/dstp;
    for (j=0; j<EOSJMAX; j++) {
        eos_t[j] = pow(10.0e0, (eos_tlo + j*tstp));
    }
    for (i=0; i<EOSIMAX; i++) {
        eos_d[i] = pow(10.0e0, (eos_dlo + i*dstp));
    }
    
    //..store the temperature and density differences and their inverses
    for (i=0; i<EOSIMAX-1; i++) {
        eos_dd[i]   = eos_d[i+1] - eos_d[i];
        eos_ddSqr[i]  = eos_dd[i]*eos_dd[i];
        eos_ddInv[i]  = 1.0e0/eos_dd[i];
        eos_ddSqrInv[i] = 1.0e0/eos_ddSqr[i];
    }
    for (j=0; j<EOSJMAX-1; j++) {
        eos_dt[j]   = eos_t[j+1] - eos_t[j];
        eos_dtSqr[j]  = eos_dt[j]*eos_dt[j];
        eos_dtInv[j]  = 1.0e0/eos_dt[j];
        eos_dtSqrInv[j] = 1.0e0/eos_dtSqr[j];
    }
    
    return;
}


void eos_helm(
                double btemp, double den, double abar, double zbar, double *pr, double *eps,
                double eos_f[EOSIMAX][EOSJMAX], double eos_fd[EOSIMAX][EOSJMAX], double eos_ft[EOSIMAX][EOSJMAX], double eos_fdd[EOSIMAX][EOSJMAX], double eos_ftt[EOSIMAX][EOSJMAX],
                double eos_fdt[EOSIMAX][EOSJMAX], double eos_fddt[EOSIMAX][EOSJMAX], double eos_fdtt[EOSIMAX][EOSJMAX], double eos_fddtt[EOSIMAX][EOSJMAX],
                double eos_dpdf[EOSIMAX][EOSJMAX], double eos_dpdfd[EOSIMAX][EOSJMAX], double eos_dpdft[EOSIMAX][EOSJMAX], double eos_dpdfdt[EOSIMAX][EOSJMAX],
                double eos_ef[EOSIMAX][EOSJMAX], double eos_efd[EOSIMAX][EOSJMAX], double eos_eft[EOSIMAX][EOSJMAX], double eos_efdt[EOSIMAX][EOSJMAX],
                double eos_xf[EOSIMAX][EOSJMAX], double eos_xfd[EOSIMAX][EOSJMAX], double eos_xft[EOSIMAX][EOSJMAX], double eos_xfdt[EOSIMAX][EOSJMAX],
                double eos_t[EOSJMAX], double eos_d[EOSIMAX], double eos_dd[EOSIMAX], double eos_ddSqr[EOSIMAX], double eos_ddInv[EOSIMAX],
                double eos_ddSqrInv[EOSIMAX], double eos_dt[EOSJMAX], double eos_dtSqr[EOSJMAX], double eos_dtInv[EOSJMAX], double eos_dtSqrInv[EOSJMAX]
                )
{
    // ******************
    // from Eos_data
    double eos_gasConstant;
    double eos_smalle;
    double eos_gamma;
    double eos_singleSpeciesA;
    double eos_singleSpeciesZ;
    double eos_eintSwitch;
    int eos_type, eos_meshMe, eos_meshNumProcs, eos_globalMe;

    double eos_largeT = 1.0e10; // used by some modified implementations of Newton-Raphson
    double eos_smallRho;

    int eos_logLevel = 700;

    // Some stuff that is only used by multiTemp implementations
    int eos_combinedTempRule = -1;

    int eos_entrEleScaleChoice = -1;

    double eos_smallEion=0.0, eos_smallEele=0.0, eos_smallErad=0.0;

  #ifdef FIXEDBLOCKSIZE
    double eos_massFr[NSPECIES*MAXCELLS];
    double eos_inOut[EOS_NUM*MAXCELLS];
  #endif

    double eos_pradScaleFactor = 1.0; //hardwired for now
  #ifdef FLLM_VAR
    int eos_pradScaleVar = FLLM_VAR;
  #else
    int eos_pradScaleVar = -1;
  #endif

    //integer, save, dimension(1:EOSMAP_NUM_ROLES, 1:2, 1:5) :: eos_mapLookup
    int eos_threadWithinBlock = 0;

    // ******************
    // from eos_helmData
    double eos_fluffDens, eos_hfetInit;

    double eos_tstpi, eos_dstpi;

    // ******************
    // from eos_vecData
    #ifdef FIXEDBLOCKSIZE
    int NROWMAX = GRID_IHI_GC * GRID_JHI_GC * GRID_KHI_GC;
    double  tempRow[NROWMAX], denRow[NROWMAX], abarRow[NROWMAX], zbarRow[NROWMAX],
            ptotRow[NROWMAX], dptRow[NROWMAX], etotRow[NROWMAX], detRow[NROWMAX], stotRow[NROWMAX],
            dedRow[NROWMAX], dstRow[NROWMAX], dsdRow[NROWMAX], dpdRow[NROWMAX],
            deaRow[NROWMAX], dezRow[NROWMAX],
            pelRow[NROWMAX], neRow[NROWMAX], etaRow[NROWMAX],
            gamcRow[NROWMAX],
            cpRow[NROWMAX], cvRow[NROWMAX];
    #endif

    // ******************
    // from eos_helmConstData
    double  pi = 3.1415926535897932384e0,
            eulercon = 0.577215664901532861e0,
            a2rad = pi / 180.0e0,
            rad2a = 180.0e0 / pi;

    double  g = 6.67259e-8,
            h       = 6.6260755e-27,
            hbar    = 0.5 * h / pi,
            e       = 4.8032068e-10,
            avo     = 6.0221367e23,
            c       = 2.99792458e10,
            kerg    = 1.380658e-16,
            kev     = 8.617385e-5,
            amu     = 1.6605402e-24,
            mn      = 1.6749286e-24,
            mp      = 1.6726231e-24,
            me      = 9.1093897e-28,
            rbohr   = hbar * hbar / (me * e * e),
            fine    = e * e / (hbar * c),
            hion    = 13.605698140e0,
            ev2erg  = 1.602e-12,
            ssol     = 5.67051e-5,
            asol    = 4.0e0 * ssol / c,
            weinlam = h * c / (kerg * 4.965114232e0),
            weinfre = 2.821439372e0 * kerg / h,
            rhonuc  = 2.342e14;

    double  msol    = 1.9892e33,
            rsol    = 6.95997e10,
            lsol    = 3.8268e33,
            mearth  = 5.9764e27,
            rearth  = 6.37e8,
            ly      = 9.460528e17,
            pc      = 3.261633e0 * ly,
            au      = 1.495978921e13,
            secyer  = 3.1558149984e7;

    double  avoinv  = 1.0e0 / avo,
            kergavo = kerg * avo,
            asoli3  = asol / 3.0e0,
            sioncon = (2.0e0 * pi * amu * kerg) / (h * h);

    // ******************
    // Local variables
    double z2bar, ytot1, ye, local_coulombMult;
    double  x1, x2, x3, x4, x5, x6, x7,
            y0, y1, y2, y3, y4,
            deni, tempi, kt,
            prad, dpraddd, dpraddt, erad, deraddd, deraddt,
            srad, dsraddd, dsraddt,
            xni, pion, dpiondd, dpiondt, eion, deiondd, deiondt,
            sion, dsiondd, dsiondt,
            pele, dpepdd, dpepdt, eele, deepdd, deepdt,
            sele, dsepdd, dsepdt,
            pres, dpresdd, dpresdt, ener, denerdt,
            entr,
            presi, chit, chid, gamc, kavoy;

     double cv, cp, etaele, xnefer,
            denerdd, dentrdd ,dentrdt;

    // For the interpolations
    int iat,jat;
    double             free,df_d,df_t,df_dd,df_tt,df_dt;
    double             xt,xd,mxt,mxd,
       si0t,si1t,si2t,si0mt,si1mt,si2mt,
       si0d,si1d,si2d,si0md,si1md,si2md,
       dsi0t,dsi1t,dsi2t,dsi0mt,dsi1mt,dsi2mt,
       dsi0d,dsi1d,dsi2d,dsi0md,dsi1md,dsi2md,
       ddsi0t,ddsi1t,ddsi2t,ddsi0mt,ddsi1mt,ddsi2mt,
       zFunc,z0,z1,z2,z3,z4,z5,z6,
       din,h3dpd,
       w0t,w1t,w2t,w0mt,w1mt,w2mt,
       w0d,w1d,w2d,w0md,w1md,w2md;
    double h3e, h3x;
    double fi[36];
    // For the coulomb corrections
    double  ktinv,dxnidd,dsdd,lami,inv_lami,lamidd,
            s0,s1,s2,s3,s4,
            plasg,plasg_inv,plasgdd,plasgdt,
            ecoul,decouldd,decouldt,pcoul,dpcouldd,dpcouldt,
            scoul,dscouldd,dscouldt;

    // Added by Calhoun for calculations for the Aprox13t network
    double  deradda,dxnida,dpionda,deionda,dsepda,deepda,decoulda,
            dsda,dsdda,lamida,plasgda,denerda,deraddz,deiondz,deepdz,
            decouldz,dsepdz,plasgdz,denerdz;
    int    bAprox13t;  // becomes true if variables for Aprox13t network are set

    double  third = 1.0e0/3.0e0,
            forth = 4.0e0/3.0e0,
            qe    = 4.8032068e-10,
            esqu  = qe * qe;


    // For the uniform background coulomb correction
    double  a1 = -0.898004e0,
            b1 = 0.96786e0,
            c1 = 0.220703e0,
            d1cc = -0.86097e0,
            e1cc = 2.5269e0,
            a2 = 0.29561e0,
            b2 = 1.9885e0,
            c2 = 0.288675e0;
    
    // For Newton-Raphson
    double eos_smallt = 1.0e-10;
    double eos_tol = 1.0e-8;
    int eos_maxNewton = 50;
    double eos_coulombMult = 1.0;
    
    int eos_coulombAbort = 1;
    int eos_forceConstantInput = 0;

    // ------------------------------------------------------------------------------

    // Initial testing of masks
    // Note that there should be things added here for entropy eventually
    bAprox13t = 0;

	// execution

    ytot1  = 1.0e0/abar;
    ye     = ytot1 * zbar;


    //  frequent combinations
    deni    = 1.0e0/den;
    tempi   = 1.0e0/btemp;
    kt      = kerg * btemp;
    ktinv   = 1.0e0/kt;
    kavoy   = kergavo * ytot1;


    //  radiation section:
    prad    = asoli3 * btemp * btemp * btemp * btemp;
    dpraddt = 4.0e0 * prad * tempi;
    dpraddd = 0.0e0;

    x1      = prad * deni;
    erad    = 3.0e0 * x1;
    deraddd = -erad*deni;
    deraddt = 4.0e0 * erad * tempi;
    //  Calhoun next two lines
    deradda = 0.0e0;
    deraddz = 0.0e0;

    srad    = (x1 + erad)*tempi;
    dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
    dsraddt = (dpraddt*deni + deraddt - srad)*tempi;


    //  ion section:
    dxnidd  = avo * ytot1;
    xni     = dxnidd * den;

    pion    = xni * kt;
    dpiondd = avo * ytot1 * kt;
    dpiondt = xni * kerg;

    eion    = 1.5e0 * pion * deni;
    deiondd = (1.5e0 * dpiondd - eion)*deni;
    deiondt = 1.5e0 * xni * kerg *deni;

    if(bAprox13t){
        dxnida  = -xni*ytot1; //Calhoun
        dpionda = dxnida * kt; //Calhoun
        deionda = 1.5e0 * dpionda*deni;  //Calhoun
        deiondz = 0.0e0; //Calhoun
    }

    //  sackur-tetrode equation for the ion entropy of
    //  a single ideal gas characterized by abar
    x2      = abar*abar*sqrt(abar) * deni*avoinv;
    y0      = sioncon * btemp;
    z0      = x2 * y0 * sqrt(y0);
    sion    = (pion*deni + eion)*tempi + kavoy*log(z0);
    dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi \
         - kavoy * deni;
    dsiondt = (dpiondt*deni + deiondt)*tempi  \
         - (pion*deni + eion) * tempi*tempi \
         + 1.5e0 * kavoy * tempi;

    //  electron-positron section:
    //  enter the table with ye*den, no checks of the input
    din = ye*den;

    //  hash locate this temperature and density
    eos_tstpi = (double)(EOSJMAX-1)/(11.0e0 - eos_tlo);
    eos_dstpi = (double)(EOSIMAX-1)/(11.0e0 - eos_dlo);

    jat = (int)((log10(btemp) - eos_tlo)*eos_tstpi) + 1;
    jat = max(1,min(jat,EOSJMAX-1))-1;
    iat = (int)((log10(din) - eos_dlo)*eos_dstpi) + 1;
    iat = max(1,min(iat,EOSIMAX-1))-1;

    //  access the table locations only once
    fi[0]  = eos_f[iat][jat];
    fi[1]  = eos_f[iat+1][jat];
    fi[2]  = eos_f[iat][jat+1];
    fi[3]  = eos_f[iat+1][jat+1];
    fi[4]  = eos_ft[iat][jat];
    fi[5]  = eos_ft[iat+1][jat];
    fi[6]  = eos_ft[iat][jat+1];
    fi[7]  = eos_ft[iat+1][jat+1];
    fi[8]  = eos_ftt[iat][jat];
    fi[9]  = eos_ftt[iat+1][jat];
    fi[10] = eos_ftt[iat][jat+1];
    fi[11] = eos_ftt[iat+1][jat+1];
    fi[12] = eos_fd[iat][jat];
    fi[13] = eos_fd[iat+1][jat];
    fi[14] = eos_fd[iat][jat+1];
    fi[15] = eos_fd[iat+1][jat+1];
    fi[16] = eos_fdd[iat][jat];
    fi[17] = eos_fdd[iat+1][jat];
    fi[18] = eos_fdd[iat][jat+1];
    fi[19] = eos_fdd[iat+1][jat+1];
    fi[20] = eos_fdt[iat][jat];
    fi[21] = eos_fdt[iat+1][jat];
    fi[22] = eos_fdt[iat][jat+1];
    fi[23] = eos_fdt[iat+1][jat+1];
    fi[24] = eos_fddt[iat][jat];
    fi[25] = eos_fddt[iat+1][jat];
    fi[26] = eos_fddt[iat][jat+1];
    fi[27] = eos_fddt[iat+1][jat+1];
    fi[28] = eos_fdtt[iat][jat];
    fi[29] = eos_fdtt[iat+1][jat];
    fi[30] = eos_fdtt[iat][jat+1];
    fi[31] = eos_fdtt[iat+1][jat+1];
    fi[32] = eos_fddtt[iat][jat];
    fi[33] = eos_fddtt[iat+1][jat];
    fi[34] = eos_fddtt[iat][jat+1];
    fi[35] = eos_fddtt[iat+1][jat+1];

    //  various differences
    xt  = max( (btemp - eos_t[jat])*eos_dtInv[jat], 0.0e0);
    xd  = max( (din - eos_d[iat])*eos_ddInv[iat], 0.0e0);
    mxt = 1.0e0 - xt;
    mxd = 1.0e0 - xd;

    //  the density and temperature basis functions
    si0t =   psi0(xt);
    si1t =   psi1(xt)*eos_dt[jat];
    si2t =   psi2(xt)*eos_dtSqr[jat];

    si0mt =  psi0(mxt);
    si1mt = -psi1(mxt)*eos_dt[jat];
    si2mt =  psi2(mxt)*eos_dtSqr[jat];

    si0d =   psi0(xd);
    si1d =   psi1(xd)*eos_dd[iat];
    si2d =   psi2(xd)*eos_ddSqr[iat];

    si0md =  psi0(mxd);
    si1md = -psi1(mxd)*eos_dd[iat];
    si2md =  psi2(mxd)*eos_ddSqr[iat];

    // the first derivatives of the basis functions
    dsi0t =   dpsi0(xt)*eos_dtInv[jat];
    dsi1t =   dpsi1(xt);
    dsi2t =   dpsi2(xt)*eos_dt[jat];

    dsi0mt = -dpsi0(mxt)*eos_dtInv[jat];
    dsi1mt =  dpsi1(mxt);
    dsi2mt = -dpsi2(mxt)*eos_dt[jat];

    dsi0d =   dpsi0(xd)*eos_ddInv[iat];
    dsi1d =   dpsi1(xd);
    dsi2d =   dpsi2(xd)*eos_dd[iat];

    dsi0md = -dpsi0(mxd)*eos_ddInv[iat];
    dsi1md =  dpsi1(mxd);
    dsi2md = -dpsi2(mxd)*eos_dd[iat];

    // the second derivatives of the basis functions
    ddsi0t =   ddpsi0(xt)*eos_dtSqrInv[jat];
    ddsi1t =   ddpsi1(xt)*eos_dtInv[jat];
    ddsi2t =   ddpsi2(xt);

    ddsi0mt =  ddpsi0(mxt)*eos_dtSqrInv[jat];
    ddsi1mt = -ddpsi1(mxt)*eos_dtInv[jat];
    ddsi2mt =  ddpsi2(mxt);

    // the free energy
    free  = h5(
         si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
         si0d,   si1d,   si2d,   si0md,   si1md,   si2md, fi);

    // derivative with respect to density
    df_d  = h5(
         si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
         dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md, fi);

    // derivative with respect to temperature
    df_t = h5(
         dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
         si0d,   si1d,   si2d,   si0md,   si1md,   si2md, fi);

    // second derivative with respect to temperature
    df_tt = h5(
         ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
         si0d,   si1d,   si2d,   si0md,   si1md,   si2md, fi);

    //  second derivative with respect to temperature and density
    df_dt = h5(
         dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
         dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md, fi);

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    si0t   =  xpsi0(xt);
    si1t   =  xpsi1(xt)*eos_dt[jat];

    si0mt  =  xpsi0(mxt);
    si1mt  =  -xpsi1(mxt)*eos_dt[jat];

    si0d   =  xpsi0(xd);
    si1d   =  xpsi1(xd)*eos_dd[iat];

    si0md  =  xpsi0(mxd);
    si1md  =  -xpsi1(mxd)*eos_dd[iat];

    //  pressure derivative with density
    dpepdd  = eos_dpdf[iat][jat]  *si0d*si0t   +   eos_dpdf[iat+1][jat]    *si0md*si0t  \
            +   eos_dpdf[iat][jat+1]  *si0d*si0mt  +   eos_dpdf[iat+1][jat+1]  *si0md*si0mt \
            +  eos_dpdft[iat][jat]    *si0d*si1t   +  eos_dpdft[iat+1][jat]    *si0md*si1t  \
            +  eos_dpdft[iat][jat+1]  *si0d*si1mt  +  eos_dpdft[iat+1][jat+1]  *si0md*si1mt \
            +  eos_dpdfd[iat][jat]    *si1d*si0t   +  eos_dpdfd[iat+1][jat]    *si1md*si0t  \
            +  eos_dpdfd[iat][jat+1]  *si1d*si0mt  +  eos_dpdfd[iat+1][jat+1]  *si1md*si0mt \
            + eos_dpdfdt[iat][jat]    *si1d*si1t   + eos_dpdfdt[iat+1][jat]    *si1md*si1t  \
            + eos_dpdfdt[iat][jat+1]  *si1d*si1mt  + eos_dpdfdt[iat+1][jat+1]  *si1md*si1mt;

    // h3dpd(iat,jat,
    //      si0t,   si1t,   si0mt,   si1mt,
    //      si0d,   si1d,   si0md,   si1md,
    //      eos_dpdf, eos_dpdft, eos_dpdfd, eos_dpdfdt);

    dpepdd  = max(ye * dpepdd,0.0e0);

    //  electron chemical potential etaele
    etaele  = eos_ef[iat][jat]    *si0d*si0t   +   eos_ef[iat+1][jat]    *si0md*si0t  \
            +   eos_ef[iat][jat+1]  *si0d*si0mt  +   eos_ef[iat+1][jat+1]  *si0md*si0mt \
            +  eos_eft[iat][jat]    *si0d*si1t   +  eos_eft[iat+1][jat]    *si0md*si1t  \
            +  eos_eft[iat][jat+1]  *si0d*si1mt  +  eos_eft[iat+1][jat+1]  *si0md*si1mt \
            +  eos_efd[iat][jat]    *si1d*si0t   +  eos_efd[iat+1][jat]    *si1md*si0t  \
            +  eos_efd[iat][jat+1]  *si1d*si0mt  +  eos_efd[iat+1][jat+1]  *si1md*si0mt \
            + eos_efdt[iat][jat]    *si1d*si1t   + eos_efdt[iat+1][jat]    *si1md*si1t  \
            + eos_efdt[iat][jat+1]  *si1d*si1mt  + eos_efdt[iat+1][jat+1]  *si1md*si1mt;

    // h3e(iat,jat,
    //                si0t,   si1t,   si0mt,   si1mt,
    //                si0d,   si1d,   si0md,   si1md);

    //  electron + positron number densities
    xnefer   = eos_xf[iat][jat]    *si0d*si0t   +   eos_xf[iat+1][jat]    *si0md*si0t  \
            +   eos_xf[iat][jat+1]  *si0d*si0mt  +   eos_xf[iat+1][jat+1]  *si0md*si0mt \
            +  eos_xft[iat][jat]    *si0d*si1t   +  eos_xft[iat+1][jat]    *si0md*si1t  \
            +  eos_xft[iat][jat+1]  *si0d*si1mt  +  eos_xft[iat+1][jat+1]  *si0md*si1mt \
            +  eos_xfd[iat][jat]    *si1d*si0t   +  eos_xfd[iat+1][jat]    *si1md*si0t  \
            +  eos_xfd[iat][jat+1]  *si1d*si0mt  +  eos_xfd[iat+1][jat+1]  *si1md*si0mt \
            + eos_xfdt[iat][jat]    *si1d*si1t   + eos_xfdt[iat+1][jat]    *si1md*si1t  \
            + eos_xfdt[iat][jat+1]  *si1d*si1mt  + eos_xfdt[iat+1][jat+1]  *si1md*si1mt;

    // h3x(iat,jat,
    //              si0t,   si1t,   si0mt,   si1mt,
    //              si0d,   si1d,   si0md,   si1md);

    //  the desired electron-positron thermodynamic quantities
    x3      = din * din;
    pele    = x3 * df_d;
    dpepdt  = x3 * df_dt;

    sele    = -df_t * ye;
    dsepdt  = -df_tt * ye;
    dsepdd  = -df_dt * ye * ye;

    eele    = ye * free + btemp * sele;
    deepdt  = btemp * dsepdt;
    deepdd  = ye*ye*df_d + btemp*dsepdd;

    if (bAprox13t) {
        dsepda  = ytot1 * (ye * df_dt * din - sele); //Calhoun
        dsepdz  = -ytot1 * (ye * df_dt * den  + df_t); //Calhoun
        deepda  = -ye * ytot1 * (free +  df_d * din) + btemp * dsepda; //Calhoun
        deepdz  = ytot1* (free + ye * df_d * den) + btemp * dsepdz; //Calhoun
    }

    //  coulomb section:
    //  initialize
    pcoul    = 0.0e0;
    dpcouldd = 0.0e0;
    dpcouldt = 0.0e0;
    ecoul    = 0.0e0;
    decouldd = 0.0e0;
    decouldt = 0.0e0;
    decoulda = 0.0e0; //Calhoun
    decouldz = 0.0e0; //Calhoun
    scoul    = 0.0e0;
    dscouldd = 0.0e0;
    dscouldt = 0.0e0;

    // Set the coulomb multiplier to a local value -- we might change it only within this call
    local_coulombMult = eos_coulombMult;

    //  uniform background corrections & only the needed parts for speed
    //  plasg is the plasma coupling parameter
    //  split up calculations below -- they all used to depend upon a redefined z
    z1        = forth * pi;
    s1        = z1 * xni;
    dsdd      = z1 * dxnidd;
    lami      = 1.0e0/pow(s1, third);
    inv_lami  = 1.0e0/lami;
    z2        = -third * lami/s1;
    lamidd    = z2 * dsdd;


    plasg     = zbar*zbar*esqu*ktinv*inv_lami;
    plasg_inv = 1.0e0/plasg;
    z3        = -plasg * inv_lami;
    plasgdd   = z3 * lamidd;
    plasgdt   = -plasg * ktinv * kerg;

    if (bAprox13t) {
        dsda     = z1 * dxnida;          //Calhoun
        lamida   = z2 * dsda / s1;      //Calhoun
        plasgda  = z3 * lamida;         //Calhoun
        plasgdz  = 2.0 * plasg/zbar;   //Calhoun
    }

    //  yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.0) {
        x4       = pow(plasg, 0.25e0);
        z4       = c1/x4;
        ecoul    = dxnidd * kt * (a1*plasg + b1*x4 + z4 + d1cc);
        pcoul    = third * den * ecoul;
        scoul    = -kavoy*(3.0e0*b1*x4 - 5.0e0*z4 \
             + d1cc*(log(plasg) - 1.0e0) - e1cc);

        y1       = dxnidd * kt * (a1 + 0.25e0*plasg_inv*(b1*x4 - z4));
        decouldd = y1 * plasgdd;
        decouldt = y1 * plasgdt + ecoul * tempi;
        dpcouldd = third * (ecoul + den * decouldd);
        dpcouldt = third * den  * decouldt;

        if (bAprox13t) {
           decoulda = y1 * plasgda - ecoul/abar;   //Calhoun
           decouldz = y1 * plasgdz;                //Calhoun
        }

        y2       = -kavoy*plasg_inv*(0.75e0*b1*x4 + 1.25e0*z4 + d1cc);
        dscouldd = y2 * plasgdd;
        dscouldt = y2 * plasgdt;
    }
       //  yakovlev & shalybkov 1989 equations 102, 103, 104
    else if (plasg < 1.0) {
       x5       = plasg * sqrt(plasg);
       y3       = pow(plasg,b2);
       z5       = c2 * x5 - third * a2 * y3;
       pcoul    = -pion * z5;
       ecoul    = 3.0e0 * pcoul * deni;
       scoul    = -kavoy*(c2*x5 - a2*(b2 - 1.0e0)/b2*y3);

       s2       = (1.5e0*c2*x5 - third*a2*b2*y3)*plasg_inv;
       dpcouldd = -dpiondd*z5 - pion*s2*plasgdd;
       dpcouldt = -dpiondt*z5 - pion*s2*plasgdt;
       decouldd = 3.0e0*dpcouldd*deni - ecoul*deni;
       decouldt = 3.0e0*dpcouldt*deni;

       // Equations for decoulda and decouldz can be added here....

       if (bAprox13t) {
           fprintf(stderr, "[eos_helm] TRAGEDY decoulda and decouldz not defined!\n");
       }

       s3       = -kavoy*plasg_inv*(1.5e0*c2*x5 - a2*(b2 - 1.0e0)*y3);
       dscouldd = s3 * plasgdd;
       dscouldt = s3 * plasgdt;
    }

    s4 = prad + pion + pele;
    x6 = s4 + pcoul*eos_coulombMult;

    // assume that NaN always compares as false in an inequality
    if ( !(x6 > 0.0) ) {
        printf('[eos_helm] Negative total pressure.\n');
        printf("%s %e %e\n", ' values: dens,temp: ',den,btemp);
        printf("%s %e %e\n", ' values: abar,zbar: ',abar,zbar);
        printf("%s %e\n", ' coulomb coupling parameter Gamma: ',plasg);

        if ( !(abar > 0.0) ) {
            printf("%s %e\n", '  However, abar is negative, abar=',abar);
            printf("%s\n", '      It is possible that the mesh is of low quality.');
            printf("%s\n", '[eos_helm] ERROR: abar is negative.');
        }

        if ( s4 > 0.0 ) {
            printf("%s %e %e\n", ' nonpositive P caused by coulomb correction: Pnocoul,Pwithcoul: ',s4,x6);

            if (eos_coulombMult > 0.0) {
                printf('  set runtime parameter eos_coulombMult to zero if plasma Coulomb corrections not important\n');
            }
            if (eos_coulombAbort) {
                printf('[eos_helm] ERROR: coulomb correction causing negative total pressure.\n');
            }
            else {
                printf('Setting coulombMult to zero for this call, eos_coulombAbort=false\n');
                local_coulombMult = 0.0;
            }
        }
        else {
            printf("Prad %e\nPion %e\nPele %e\nPcoul %e\nPtot %e\ndf_d %e\n", prad, pion, pele, pcoul*eos_coulombMult, x6, df_d);
            printf('[eos_helm] ERROR: negative total pressure.\n');
        }

    }

    pcoul    = pcoul * local_coulombMult;
    dpcouldd = dpcouldd * local_coulombMult;
    dpcouldt = dpcouldt * local_coulombMult;

    ecoul    = ecoul * local_coulombMult;
    decouldd = decouldd * local_coulombMult;
    decouldt = decouldt * local_coulombMult;

    scoul    = scoul * local_coulombMult;
    dscouldd = dscouldd * local_coulombMult;
    dscouldt = dscouldt * local_coulombMult;

    pres    = prad    + pion    + pele   + pcoul;
    ener    = erad    + eion    + eele   + ecoul;
    entr    = srad    + sion    + sele   + scoul;

    dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd;
    dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt;

    denerdd = deraddd + deiondd + deepdd + decouldd;
    denerdt = deraddt + deiondt + deepdt + decouldt;

    dentrdd = dsraddd + dsiondd + dsepdd + dscouldd;
    dentrdt = dsraddt + dsiondt + dsepdt + dscouldt;

    //  form gamma_1
    presi = 1.0e0/pres;
    chit  = btemp*presi * dpresdt;
    chid  = dpresdd * den*presi;
    x7     = pres * deni * chit/(btemp * denerdt);
    gamc  = chit*x7 + chid;
    cv    = denerdt;
    cp    = cv*gamc/chid;

    //  store the output -- note that many of these are not used by the calling program!
    // ptotRow(j)   = pres;   //used by Eos as EOS_PRES = PRES_VAR
    // etotRow(j)   = ener;   //used by Eos as EOS_EINT = EINT_VAR
    // stotRow(j)   = entr;   //this is entropy, used by Eos as EOS_ENTR (masked)

    // dpdRow(j)    = dpresdd;  // used as EOS_DPD
    // dptRow(j)    = dpresdt;  // used as EOS_DPT ALWAYS used by MODE_DENS_PRES in Eos.F90
    //
    // dedRow(j)    = denerdd;  // used as EOS_DED
    // detRow(j)    = denerdt;  // used as EOS_DET  ALWAYS used by MODE_DENS_EI in Eos.F90
    //
    // if (bAprox13t) {
    //     denerda = deradda + deionda + deepda + decoulda;  //Calhoun
    //     denerdz = deraddz + deiondz + deepdz + decouldz;  //Calhoun
    //     deaRow(j)    = denerda;  //Calhoun EOS_DEA
    //     dezRow(j)    = denerdz;  //Calhoun EOS_DEZ
    // }
    //
    // dsdRow(j)    = dentrdd;  // used as EOS_DSD
    // dstRow(j)    = dentrdt;  // used as EOS_DST
    //
    // pelRow(j)   = pele;     // used as EOS_PEL
    //
    // neRow(j)    = xnefer;   // used as EOS_NE
    // etaRow(j) = etaele;     // used as EOS_ETA
    //
    // gamcRow(j)   = gamc;    // used as EOS_GAMC = GAMC_VAR
    //
    // cvRow(j)     = cv;      // EOS_CV
    // cpRow(j)     = cp;      // EOS_CP

	*pr = pres;
	*eps = ener;
    return;
}

int main() {
    
    // Arrays declaration:
    double eos_dt[EOSJMAX], eos_dtSqr[EOSJMAX], eos_dtInv[EOSJMAX], eos_dtSqrInv[EOSJMAX], eos_t[EOSJMAX];
    double eos_dd[EOSIMAX], eos_ddSqr[EOSIMAX], eos_ddInv[EOSIMAX], eos_ddSqrInv[EOSIMAX], eos_d[EOSIMAX];
    
    double  eos_f[EOSIMAX][EOSJMAX], eos_fd[EOSIMAX][EOSJMAX], eos_ft[EOSIMAX][EOSJMAX], eos_fdd[EOSIMAX][EOSJMAX],
            eos_ftt[EOSIMAX][EOSJMAX], eos_fdt[EOSIMAX][EOSJMAX], eos_fddt[EOSIMAX][EOSJMAX],
            eos_fdtt[EOSIMAX][EOSJMAX], eos_fddtt[EOSIMAX][EOSJMAX],
            eos_dpdf[EOSIMAX][EOSJMAX], eos_dpdfd[EOSIMAX][EOSJMAX], eos_dpdft[EOSIMAX][EOSJMAX], eos_dpdfdt[EOSIMAX][EOSJMAX],
            eos_ef[EOSIMAX][EOSJMAX], eos_efd[EOSIMAX][EOSJMAX], eos_eft[EOSIMAX][EOSJMAX], eos_efdt[EOSIMAX][EOSJMAX],
            eos_xf[EOSIMAX][EOSJMAX], eos_xfd[EOSIMAX][EOSJMAX], eos_xft[EOSIMAX][EOSJMAX], eos_xfdt[EOSIMAX][EOSJMAX];
    
	double btemp=2.0e8, den=1.0e7;
	double abar=1.0, zbar=1.0;
	double pr, eps;
//    int ix,jx;

    // Reading the table and writing into arrays
    eos_init(eos_f, eos_fd, eos_ft, eos_fdd, eos_ftt,
             eos_fdt, eos_fddt, eos_fdtt, eos_fddtt,
             eos_dpdf, eos_dpdfd, eos_dpdft, eos_dpdfdt,
             eos_ef, eos_efd, eos_eft, eos_efdt,
             eos_xf, eos_xfd, eos_xft, eos_xfdt,
             eos_t, eos_d, eos_dd, eos_ddSqr, eos_ddInv,
             eos_ddSqrInv, eos_dt, eos_dtSqr, eos_dtInv, eos_dtSqrInv);
    
    eos_helm(btemp, den, abar, zbar, &pr, &eps,
             eos_f, eos_fd, eos_ft, eos_fdd, eos_ftt,
             eos_fdt, eos_fddt, eos_fdtt, eos_fddtt,
             eos_dpdf, eos_dpdfd, eos_dpdft, eos_dpdfdt,
             eos_ef, eos_efd, eos_eft, eos_efdt,
             eos_xf, eos_xfd, eos_xft, eos_xfdt,
             eos_t, eos_d, eos_dd, eos_ddSqr, eos_ddInv,
             eos_ddSqrInv, eos_dt, eos_dtSqr, eos_dtInv, eos_dtSqrInv);
    
    printf("d=%21.15e, T=%21.15e, Pressure = %21.15e, Energy = %21.15e\n", den, btemp, pr, eps);
//    for (ix=-10;ix<12;ix++){
//        den = pow(10.0, ix);
//        for (jx=4;jx<12;jx++){
//            btemp = pow(10.0, jx);
//            eos_helm(btemp, den, abar, zbar, &pr, &eps);
//            printf("d=%e, T=%e, Pressure = %e, Energy = %e\n", den, btemp, pr, eps);
//        }
//    }

	return 0;
}
