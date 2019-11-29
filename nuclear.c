//
//  nuclear.c
//  HARM2D
//
//  Created by Alexander Tchekhovskoy on 9/8/15.
//  Copyright (c) 2015 Home. All rights reserved.
//
//  Edited by Danat Issa on 10/17/19

#include "decs.h"
#include "nuclear.h"

#if(DONUCLEAR)
double D4rodrigo(double eta, double Xn, double Xp);
double D5rodrigo(double eta, double Xn, double Xp);
double F4m(double x);
double F4p(double x);
double F5m(double x);
double F5p(double x);

#if(DOHELM)
void nuc_evol_helm (double pr[NPR], double Dt, int i, int j, int k, int was_floor_activated, int n) {
    
    //all sorts of nuclear physics
    double Xalpha, Xn, Xp, Xnp, Xfloor, Xamb, rho, rhofloor, rhotot, ug;
    double Xalphanew, Xalpha_0, dXalpha;
    double fac;
    double T, Ye;
    const double T_unit = compute_Tunit();
    struct of_geom geom;
    double ucon[NDIM], X[NDIM];
    double etae, eV_cgs, MeV_cgs, T_MeV, rho_10, T_10, Xwb;
    double dqalpha;
    double G, Q, dG, dQ;
    double expmtaunu;
    double xxx;
    double r, th, phi, ufloor;
    //int ind, ind_max;            //for subcycling
    //double frac, mult, subcyc;
    int ind;                       //for implicit update of uu
    double Fnr, dFnr, a_nr, b_nr, c_nr, d_nr, e_nr;
    
    //***** Part 1: Compute mass fractions: Xnp, Xalpha
    
    // If the abundances are negative, set them to zero
    if (pr[RHONP] < 0.)      pr[RHONP] = 0.;
    if (pr[RHOALPHA] < 0.)   pr[RHOALPHA] = 0.;
    if (pr[RHOFLOOR] < 0.)   pr[RHOFLOOR] = 0.;
    
    // Re-normalize the abundances
    fac = 1.0 / (pr[RHONP] + pr[RHOALPHA] + pr[AMB] + SMALL);
    pr[RHOALPHA] *= fac;
    pr[AMB]      *= fac;
    pr[RHONP]    *= fac; //this does not matter since not used below
    
    // Mass fractions
    Xalpha_0 = pr[RHOALPHA]; // / (rho+SMALL);
    Xalpha = Xalpha_0;
    Xamb = pr[AMB]; // / (rho+SMALL);
    
    // Get the density, internal energy and electron fraction
    rho = pr[RHO];
    ug = pr[UU];
    Ye = pr[YE];
    
    // If, for some reason, the density is negative, set it to zero
    if (rho < 0.0) rho = SMALL;
    
    // convert into cgs from code units
//    rho *= conv_dens_CODE2CGS;
//    ug *= conv_ener_CODE2CGS * conv_dens_CODE2CGS;
    
    //floor+ambient density
    //rhofloor = pr[RHOFLOOR];
    
    //***** End of Part 1
    
    // Danat: given   rho, ug, Ye
    //        find    T, Xa, elaele
    double xener = ug / rho;
    xener = xener - 6.8e18 / 9e20 * Xalpha_0; // already in code units
    eos_mode_dens_ener_nuclear_nucevol (xener, rho, &T, Ye, &Xamb, &Xn, &Xp, &Xalpha, &etae); // inputs in code units, T is given in K
    xener = xener + 6.8e18 / 9e20 * Xalpha;   // in code units as well
    
    //Xalpha + dXalpha = min(2Ye,2-2Ye) * (1-min(1,Xwb))
    rho *= conv_dens_CODE2CGS;                  // rho from code units to g/cm^3
    eV_cgs = 1.6021772e-12;                     //erg/eV
    MeV_cgs = 1e-6 * kerg / eV_cgs;
    T_MeV = T * MeV_cgs;                        // T in units of MeV
    rho_10 = rho * 1.e-10;                      // rho in units of 1e10 g/cm^3
    T_10 = T * 1.e-10;                          // T in units of 1e10 K
    
    //compute u^t = ucon[0] = dt/dtau
    get_geometry (n, i, j, k, CENT, &geom);
    ucon_calc (pr, &geom, ucon);
    
    //get the true density floor
    coord(n, i, j, k, CENT, X);
    r = X[1]; th = X[2]; phi = X[3];
    get_rho_u_floor (r, th, phi, &rhofloor, &ufloor);
    
    //update the mass fractions
    pr[RHONP]    = Xn + Xp; //*rho;
    pr[RHOALPHA] = Xalpha; //*rho;
    pr[AMB]      = Xamb; //*rho;
    pr[RHOFLOOR] = rhofloor;
    
    //heat per unit mass converted to code units from cgs
//    dqalpha = 6.8e18 / 9e20 * (Xalpha - Xalpha_0);  //heating in a time step [erg/g] / c^2, in code units now
//    // Danat: mass fractions freeze anyway below this T
//    if (T_10 <= 0.5 || rho <= 10. * rhofloor) {
//        dqalpha = 0.0;
//    }
    
    expmtaunu = exp (- rho / 1.0e11);
    
    G = 0.0;
    if (rho > 10.0 * rhofloor) G = 0.22 * pow (T_10, 5.0) * D4 (etae, Xn, Xp) * expmtaunu; //[1/s]

    Q = 0.0;
    if (rho > 10.0 * rhofloor) Q = -8.9e17 * pow (T_10, 6.0) * D5 (etae, Xn, Xp) * expmtaunu; //[erg/g/s]
    
    //convert G and Q from cgs to code units and from rates to increments
    G /= 6.77e4;
    Q /= 6.08e25;  //[code units of energy per unit mass per unit time]
    
    //save G and Q to an array that's going to be written out to disk
//    G_global[i][j][k] = G;
//    Q_global[i][j][k] = Q;
//    qalpha_global[i][j][k] = dqalpha * ucon[0] / Dt;
    //convert per unit mass heating rate into volumetric heating rate
//    Q *= rho;
//    dqalpha *= rho;
    
    //dx/dtau * dtau/dt = dx/dt, where ucon[0] = u^t = dt/dtau
    dG = G * Dt / ucon[0];
    dQ = Q * Dt / ucon[0];
    
    pr[YE] += dG;
    
    if (pr[YE] < 0.0) {
        pr[YE] = SMALL;
    }
    
    if (pr[YE] > 1.0) {
        pr[YE] = 1.0;
    }
    
    rho *= conv_dens_CGS2CODE; // in code units
    if (T_10 > 0.5) {
        xener = (xener + dQ + 6.8e18 / 9e20 * (Xalpha - Xalpha_0)) - 6.8e18 / 9e20 * Xalpha; // in code units
        eos_mode_dens_ener_nuclear_nucevol (xener, rho, &T, Ye, &Xamb, &Xn, &Xp, &Xalpha, &etae);
        xener = xener + 6.8e18 / 9e20 * Xalpha; // in code units
    }
    
    pr[UU] = xener * rho; // in code units
    
    if (pr[UU] < ufloor) pr[UU] = ufloor;
    
    
//    dqalpha = 6.8e18 / 9e20 * (Xalpha - Xalpha_0);
//    qalpha_global[i][j][k] = dqalpha * ucon[0] / Dt;
    
    // Renormalize once again before leaving
    fac = 1.0 / (Xn + Xp + Xalpha + Xamb + SMALL);
    pr[RHONP]      = (Xn + Xp) * fac;
    pr[RHOALPHA]   = Xalpha * fac;
    pr[AMB]        = Xamb * fac;
    
}
#endif

void nuc_evol (double pr[NPR], double Dt, int i, int j, int k, int was_floor_activated, int n) {
    
    //all sorts of nuclear physics
    double Xalpha, Xn, Xp, Xnp, Xfloor, Xamb, rho, rhofloor, rhotot, ug;
    double Xalphanew, dXalpha;
    double fac;
    double T, Ye;
    const double T_unit = compute_Tunit();
    struct of_geom geom;
    double ucon[NDIM], X[NDIM];
    double etae, eV_cgs, MeV_cgs, T_MeV, rho_10, T_10, Xwb;
    double dqalpha;
    double G, Q, dG, dQ;
    double expmtaunu;
    double xxx;
    double r, th, phi, ufloor;
    int ind;                       //for implicit update of uu
    double Fnr, dFnr, a_nr, b_nr, c_nr, d_nr, e_nr;

    
    //***** Part 1: Compute mass fractions: Xnp, Xalpha
    
    // If the abundances are negative, set them to zero
    if (pr[RHONP] < 0.)      pr[RHONP] = 0.;
    if (pr[RHOALPHA] < 0.)   pr[RHOALPHA] = 0.;
    if (pr[RHOFLOOR] < 0.)   pr[RHOFLOOR] = 0.;

    // Re-normalize the abundances
    fac = 1.0 / (pr[RHONP] + pr[RHOALPHA] + pr[AMB] + SMALL);
    pr[RHOALPHA] *= fac;
    pr[AMB]      *= fac;
    pr[RHONP]    *= fac; //this does not matter since not used below
    
    // Mass fractions
    Xalpha = pr[RHOALPHA]; // / (rho+SMALL);
    Xamb = pr[AMB]; // / (rho+SMALL);
    
    // Get the density, internal energy and electron fraction
    rho = pr[RHO];
    ug = pr[UU];
    Ye = pr[YE];
    
    // If, for some reason, the density is negative, set it to zero
    if (rho < 0.0) rho = 0.0;

    //floor+ambient density
    //rhofloor = pr[RHOFLOOR];
    
    //***** End of Part 1

    // Compute temperature accounting for a mixture of gas and neutrinos & compute degeneracy
    T = compute_temperature (rho, (gam - 1.0) * ug, Ye); // multiply by T_unit to get it in K
    if (T < SMALL) T = SMALL; //to avoid division by zero later on
    etae = compute_degeneracy (rho, T, Ye);

    //Xalpha + dXalpha = min(2Ye,2-2Ye) * (1-min(1,Xwb))
    eV_cgs = 1.6021772e-12;                      // erg/eV
    MeV_cgs = 1.0e-6 * kerg / eV_cgs;            // MeV to cgs
    T_MeV = T * T_unit * MeV_cgs;                // T in units of MeV
    rho_10 = rho * conv_dens_CODE2CGS * 1.0e-10; // rho in units of 1e10 g/cm^3
    T_10 = T * T_unit * 1.0e-10;                 // T in units of 1e10 K
    Xwb = 15.58 * pow (T_MeV, 1.125) * pow (rho_10, -0.75) * exp (-7.074 / T_MeV);

    if (Xamb < 1 && T_10 > 0.5) {
        Xalphanew = MY_MIN (2.0 * Ye, 2.0 * (1.0 - Ye)) * (1. - MY_MIN (1.0, Xwb)) - Xamb;
        Xalphanew = MY_MAX (Xalphanew, 1.0e-10);
        dXalpha = Xalphanew - Xalpha;
        //update Xalpha, Xn, and Xp
        Xalpha = Xalphanew;

        Xp = Ye - 0.5*Xalpha - Xamb;
        Xp = MY_MAX (Xp, 1.e-10);

        Xn = 1. - Xp - Xalpha - Xamb;
        Xn = MY_MAX (Xn, 1.e-10);

        //renormalize abundances
        fac = Xn + Xp + Xalpha + Xamb;
        Xn     /= fac;
        Xp     /= fac;
        Xalpha /= fac;
        Xamb   /= fac;
    }

    else {
        //abundances are frozen
        Xp = Ye - 0.5*Xalpha - Xamb;
        Xp = MY_MAX (Xp, 1.e-10);

        Xn = 1. - Xp - Xalpha - Xamb;
        Xn = MY_MAX (Xn, 1.e-10);

        dXalpha = 0.;

        //renormalize abundances
        fac = Xn + Xp + Xalpha + Xamb;
        Xn     /= fac;
        Xp     /= fac;
        Xalpha /= fac;
        Xamb   /= fac;
    }

    // Compute u^t = ucon[0] = dt/dtau
    get_geometry (n, i, j, k, CENT, &geom);
    ucon_calc (pr, &geom, ucon);

    // Get the true density floor
    coord(n, i, j, k, CENT, X);
    r = X[1]; th = X[2]; phi = X[3];
//    get_rho_u_floor (r, th, phi, &rhofloor, &ufloor);

    // Update the mass fractions
    pr[RHONP]    = Xn + Xp;     // *rho;
    pr[RHOALPHA] = Xalpha;      // *rho;
    pr[AMB]      = Xamb;        // *rho;
    pr[RHOFLOOR] = rhofloor;

    // Heat per unit mass converted to code units from cgs
    dqalpha = 6.8e18 / 9e20 * dXalpha;     //heating in a time step [erg/g]
    // dqalpha /= 9e20;                    //erg/g = c^2
    
    if (T_10 <= 0.5 || rho <= 10.0 * rhofloor) dqalpha = 0.0;

    expmtaunu = exp (-rho_10 * 0.1);
    
    G = 0.0;
    if (rho > 10.0 * rhofloor) G = 0.22 * pow (T_10, 5.) * D4 (etae, Xn, Xp) * expmtaunu; // [1/s]
    
    // per unit mass dissipation rate, erg/g/s
    // xxx = D5 (etae, Xn, Xp);
    
    Q = 0.0;
    if (rho > 10.0 * rhofloor) Q = -8.9e17 * pow (T_10, 6.) * D5 (etae, Xn, Xp) * expmtaunu; // [erg/g/s]
    
    // Convert G and Q from cgs to code units and from rates to increments
    G /= 6.77e4;    // 1 / [rg / c] = c^3 / (G M_bh)
    Q /= 6.08e25;   // 1 / (G M_bh / c^5) [code units of energy per unit mass per unit time]
    
    // DANAT: leave this commented for now
    // save G and Q to an array that's going to be written out to disk
    //  G_global[i][j][k] = G;
    //  Q_global[i][j][k] = Q;
    //  qalpha_global[i][j][k] = dqalpha*ucon[0]/Dt;
    
    // Convert per unit mass heating rate into volumetric heating rate
    Q       *= rho;
    dqalpha *= rho;
    
    // dx/dtau * dtau/dt = dx/dt, where ucon[0] = u^t = dt/dtau
    dG = G * Dt / ucon[0];
    dQ = Q * Dt / ucon[0];

    // Apply nuclear heating directly to internal energy
    // pr[UU] += dqalpha + dQ;

    pr[YE] += dG;

    //implicit update of the internal energy when qalpha != 0.
    if (T_10 > 0.5) {

        Xalpha = Xalphanew - dXalpha;
        // qalpha_global[i][j][k] = 0.;

        a_nr = (7.5657e-15 * T_unit * T_unit * conv_dens_CGS2CODE) * (T_unit * T_unit / 9e+20);
        b_nr = rho * 6.8e18 / 9e20;
        c_nr = ug + dQ - b_nr * Xalpha; //RHS
        d_nr = MY_MIN (2. * Ye, 2. * (1. - Ye));
        e_nr = 15.58 * pow(rho_10, -0.75);


        //Newton-Raphson loop
        for (ind = 0; ind <= 50; ind++) {
            
            T     = compute_temperature (rho, (gam - 1) * ug, Ye);
            T_MeV = T * T_unit * MeV_cgs;
            Xwb   = e_nr * pow (T_MeV, 1.125) * exp(-7.074 / T_MeV);
            Xalphanew = d_nr * (1. - MY_MIN (1., Xwb)) - Xamb;
            Xalphanew = MY_MAX (Xalphanew, 1.e-10);

            Fnr = ug - b_nr * Xalphanew - c_nr; //function to zero
            if (Xwb >= 1.) {
                //Xalpha is constant and zero, update ug with cooling and return
                pr[UU] += dQ;
                break;
            }

            //derivative
            dFnr = 1. + b_nr * d_nr * (9. / 8. + 7.074 / T_MeV) * Xwb / (ug + a_nr * pow (T, 4.) / (gam - 1.));

            //DEBUG:
            //if (i==15 && j==5 && mpi_rank==21)
            //  fprintf(stdout,"ug, Fnr/dFnr, rank, ind = %20.12e, %20.12e, %d, %d\n", ug, Fnr/dFnr,mpi_rank,ind);
            //fprintf(stdout,"i,j,k = %d, %d, %d\n", i,j,k);

            if (fabs (Fnr / dFnr) < 1e-8 * ug) {

                //converged, update ug, qalpha, and abundances
                pr[UU] = ug;

                dqalpha = b_nr / rho * (Xalphanew - Xalpha);
                // qalpha_global[i][j][k] = dqalpha * ucon[0] / Dt;

                Xalpha = Xalphanew;
                Xp = Ye - 0.5 * Xalpha - Xamb;
                Xp = MY_MAX (Xp, 1.e-10);
                Xn = 1. - Xp - Xalpha - Xamb;
                Xn = MY_MAX (Xn, 1.e-10);

                fac = Xn + Xp + Xalpha + Xamb;
                Xn     /= fac;
                Xp     /= fac;
                Xalpha /= fac;
                Xamb   /= fac;
                
                pr[RHONP]      = Xn + Xp;
                pr[RHOALPHA]   = Xalpha;
                pr[AMB]        = Xamb;

                break;

            }

            ug = ug - Fnr / dFnr; //update ug

            if (ug < ufloor) {
                pr[UU] = ufloor;
                break;
            }

            if (ind == 50) {
                fprintf (stderr, "N-R for ug did not converge\n");
                //fprintf(stderr, "rho = %11.3e\n", rho*rho_unit);
                exit (2345);
            }
        }
    }
    else {
        //same as original, but dqalpha=0 for T_10 < 0.5
        pr[UU] += dqalpha + dQ;

    } //end implicit update logical block

    //  //subcycle if heating/cooling increment is too large relative to uu (obsolete)
    //
    //  mult   = fabs((dQ + dqalpha)/ug);  // mult = Dt / dt_heat
    //  subcyc = 0.05;                     // fraction of dt_heat we want to subcycle over
    //
    //  if (mult > subcyc) {
    //
    //    frac = subcyc/mult;  //fraction of dt_heat we're stepping forward
    //    pr[i][j][k][UU] += frac*(dqalpha + dQ);
    //    pr[i][j][k][YE] += frac*dG;
    //
    //    ind_max = floor(mult/subcyc) + 1; // integer number of dt_heat steps inside Dt, plus extra bit
    //    for (ind=2; ind <= ind_max; ind++) {
    //      //loop over rest of increment in units of subcyc, except for last step
    //      if (ind == ind_max) frac = 1. - (subcyc/mult)*floor(mult/subcyc); //last bit is remainder of Dt/(subcyc*dt_heat)
    //
    //      //recompute temperature, mass fractions, and source terms, keeping constant density
    //      ug   = pr[i][j][k][UU];
    //      Ye   = pr[i][j][k][YE];
    //      T    = compute_temperature(rho, (gam-1)*ug, Ye);
    //      etae = compute_degeneracy(rho, T, Ye);
    //
    //      T_MeV = T*T_unit*MeV_cgs;
    //      T_10  = T*T_unit*1.e-10;
    //      Xwb   = 15.58*pow(T_MeV,1.125)*pow(rho_10,-0.75)*exp(-7.074/T_MeV);
    //      if (Xamb < 1 && T_10 > 0.5) {
    //        Xalphanew = MY_MIN(2.*Ye,2.*(1.-Ye)) * (1.-MY_MIN(1.,Xwb)) - Xamb;
    //        Xalphanew = MY_MAX(Xalphanew, 1.e-10);
    //        dXalpha   = Xalphanew - Xalpha;
    //        Xalpha    = Xalphanew;
    //
    //        Xp = Ye - 0.5*Xalpha - Xamb;
    //        Xp = MY_MAX(Xp, 1.e-10);
    //        Xn = 1. - Xp - Xalpha - Xamb;
    //        Xn = MY_MAX(Xn, 1.e-10);
    //
    //      } else {
    //        Xp = Ye - 0.5*Xalpha - Xamb;
    //        Xp = MY_MAX(Xp, 1.e-10);
    //        Xn = 1. - Xp - Xalpha - Xamb;
    //        Xn = MY_MAX(Xn, 1.e-10);
    //        dXalpha = 0.;
    //
    //      } //end mass fraction computation
    //      fac = Xn + Xp + Xalpha + Xamb;
    //      Xn     /= fac;
    //      Xp     /= fac;
    //      Xalpha /= fac;
    //      Xamb   /= fac;
    //
    //      get_geometry(i,j,k,CENT,&geom) ;
    //      ucon_calc(pr[i][j][k], &geom, ucon) ;
    //      get_phys_coord(i,j,k,&r,&th,&phi);
    //      get_rho_u_floor(r, th, phi, &rhofloor, &ufloor);
    //
    //      dqalpha = 6.8e18*dXalpha/9e20;
    //      if (T_10 <= 0.5 || rho <= 10.*rhofloor) dqalpha = 0.;
    //      G = 0.;
    //      if (rho > 10.*rhofloor)
    //        G = 0.22 * pow(T_10,5.) * D4(etae,Xn,Xp) * expmtaunu;
    //      xxx = D5(etae,Xn,Xp);
    //      Q = 0.;
    //      if (rho > 10.*rhofloor)
    //        Q = -8.9e17 * pow(T_10,6.) * xxx * expmtaunu;
    //      G /= 6.77e4;
    //      Q /= 6.08e25;
    //      Q *= rho;
    //      dqalpha *= rho;
    //      dG = G * Dt / ucon[0];
    //      dQ = Q * Dt / ucon[0];
    //
    //      pr[i][j][k][UU] += frac*(dqalpha + dQ);
    //      pr[i][j][k][YE] += frac*dG;
    //
    //    } //end subcycle loop
    //
    //    pr[i][j][k][RHONP]    = (Xn+Xp); //abundances are stored at end of step
    //    pr[i][j][k][RHOALPHA] = Xalpha;  //  for consistency
    //    pr[i][j][k][AMB]      = Xamb;
    //    pr[i][j][k][RHOFLOOR] = rhofloor;
    //
    //  } else {
    //    //original
    //    //apply nuclear heating directly to internal energy
    //    pr[i][j][k][UU] += dqalpha + dQ;
    //
    //    pr[i][j][k][YE] += dG;
    //
    //  } //end subcycle logical block

}

double compute_dq_unit () {
    const double mneutron_cgs = 1.6749286e-24;
    const double mneutron = mneutron_cgs/Mbh_cgs;
    const double G = 6.67259e-8;
    const double c = 2.99792458e10;
    const double hbar = 1.05457266e-27;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    const double rho_unit = Mbh_cgs / (rg_cgs*rg_cgs*rg_cgs);
    const double mass_unit = Mbh_cgs;  //[g], BH mass
    //const double time_unit = rg_cgs/c; //[s], light crossing time of r_g
    const double energy_unit = mass_unit*c*c; //[erg], energy unit
    const double dq_unit = energy_unit / (mass_unit);
    return (dq_unit);
}

double compute_t_unit () {
    const double G = 6.67259e-8;
    const double c = 2.99792458e10;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    const double t_unit = rg_cgs/c;
    return (t_unit);
}

double compute_M_unit () {
    return (Mbh_cgs);
}


double compute_Tunit () {
    const double c = 2.99792458e10;
    const double mneutron_cgs = 1.6749286e-24;
    const double T_unit = mneutron_cgs*c*c/kerg;
    return (T_unit);
}

double compute_L_unit () {
    const double c = 2.99792458e10;
    const double G = 6.67259e-8;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    return (rg_cgs);
}

//equation of state stuff: rho, p -> T
// Danat: multiply T by T_unit = kb / (m_n c^2) to obtain it in K
double compute_temperature (double rho, double p, double Ye) {
    //COMPUTE THE TEMPERATURE (no nuclear recombination yet)
    const double mneutron_cgs = 1.6749286e-24;
    const double c = 2.99792458e10;
    const double G = 6.67259e-8;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    const double hbar = 1.05457266e-27;
    const double lambdac_cgs = hbar / (mneutron_cgs*c);
    const double mneutron = mneutron_cgs/Mbh_cgs;
    const double lambdac = lambdac_cgs/rg_cgs;
    double atemp = (M_PI*M_PI/45.) * (1./p) * mneutron * pow(lambdac,-3.);
    double btemp = (1. + Ye) * (rho / p);
    double Fzero, dFzero;

    //initial guess assuming radiation-pressure-dominance
    double x = pow(atemp,-0.25);

    double prec_newton = 1e-6;
    int ind;
    const int max_iter = 50;

    for (ind=1; ind < max_iter; ind++) {
        Fzero  =       atemp * pow (x, 4.0) + btemp * x - 1.0;
        dFzero = 4.0 * atemp * pow (x, 3.0) + btemp;

        x = x - Fzero / dFzero;
        if (fabs (Fzero / dFzero) < prec_newton * x) break;
    }
    
    if (max_iter == ind) {
        fprintf (stderr, "Cool: N-R for temperature did not converge\n" );
        fprintf (stderr, "rho = %11.3e\n", rho * Mbh_cgs / pow (rg_cgs, 3.0));
        fprintf (stderr, "p   = %11.3e\n", p * Mbh_cgs * c * c / pow (rg_cgs, 3.0));
        fprintf (stderr, "Ye  = %11.3e\n", Ye);
        exit (2345);
    }
    return (x);
}

//accepts temperature in code units T = k Tcgs / mneutron_cgs c^2
double compute_degeneracy (double rho, double T, double Ye) {
    //Compute degeneracy parameter
    const double mneutron_cgs = 1.6749286e-24;
    const double mneutron = mneutron_cgs/Mbh_cgs;
    const double G = 6.67259e-8;
    const double c = 2.99792458e10;
    const double hbar = 1.05457266e-27;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    const double rho_unit = Mbh_cgs / (rg_cgs*rg_cgs*rg_cgs);
    double ANN, BOB;
    const double charlie = (M_PI*M_PI*M_PI)*(M_PI*M_PI*M_PI)/27.;
    double pf_cgs = hbar * pow(3.*M_PI*M_PI*Ye*rho*rho_unit/mneutron_cgs,1./3.);
    double T_cgs = T*mneutron_cgs*c*c/kerg;
    double pfcokT = pf_cgs * c / (kerg * T_cgs);
    double pfcokT3 = (pfcokT*pfcokT)*pfcokT;
    double pfcokT6 = pfcokT3*pfcokT3;
    double etae;

    if (T_cgs > 1.1604519308e+10) {
        ANN = 0.5*pfcokT3;
        BOB = sqrt( 0.25*pfcokT6 + charlie );
        etae = pow(ANN+BOB,1./3.) - pow(BOB-ANN,1./3.);
    }
    else {
        etae = 0.0;
    }
    return (etae);
}

double D4sasha (double eta, double Xn, double Xp);
double D5sasha (double eta, double Xn, double Xp);

double D4(double eta, double Xn, double Xp) {
    double a, b, diff;
//    a = D4sasha(eta,Xn,Xp);
    b = D4rodrigo (eta,Xn,Xp);
//    diff = 2.*fabs(a-b)/(fabs(a)+fabs(b)+SMALL);
//    if(diff>1e-12) {
//        fprintf(stderr, "D4diff = %g; s = %21.15g, r = %21.15g, eta = %g, Xn = %g, Xp = %g\n", diff, a, b, eta, Xn, Xp);
//    }
    return (b);
}

double D5 (double eta, double Xn, double Xp) {
    double a, b, diff;
//    a = D5sasha(eta,Xn,Xp);
    b = D5rodrigo (eta, Xn, Xp);
//    diff = 2.*fabs(a-b)/(fabs(a)+fabs(b)+SMALL);
//    if(diff>1e-12) {
//        fprintf(stderr, "D5diff = %g; s = %21.15g, r = %21.15g, eta = %g, Xn = %g, Xp = %g\n", diff, a, b, eta, Xn, Xp);
//        fprintf(stderr, "s: F5m = %g; F5p = %21.15g, r: F5m = %g, F5p = %g\n",
//                F5neg(-eta), F5(eta), F5m(eta), F5p(eta));
//    }
    return (b);
}

double D4sasha (double eta, double Xn, double Xp) {
    double res;
    res = (Xn*F4(-eta)-Xp*F4(eta))/F4(0);
    return (res);
}

double D5sasha (double eta, double Xn, double Xp) {
    double res;
    res = (Xn*F5(-eta)+Xp*F5(eta))/F5(0);
    return (res);
}

double F4 (double eta) {
    double res, eta2;
    if (eta<=0) {
        return(F4neg(eta));
    }
    eta2 = eta*eta;
    res = eta*(45.45757581586780404367+eta2*(6.57973626739290574589 + 0.2 * eta2));
    res += F4neg(-eta);
    return( res );
}

double F5(double eta)
{
    double res, eta2;
    if (eta<=0) {
        return(F5neg(eta));
    }
    eta2 = eta*eta;
    res = 236.53226191138442498363 //4.63758915529921859011
        + eta2*(113.64393953966951010918
                    + eta2*(8.22467033424113218236 + eta2*0.16666666666666666667)
                 );
    res -= F5neg(-eta);
    return( res );
}

double F4neg (double eta) {
    double res;
    double y, y2, y3;
    
    y  = exp (eta);
    y2 = y * y;
    y3 = y2 * y;
    
    res = 24. * (y - y2 * 0.03125 + y3 * 0.00411522633744855967);
    
    return (res);
}

double F5neg (double eta) {
    double res;
    double y, y2, y3;
    
    y  = exp (eta);
    y2 = y * y;
    y3 = y2 * y;
    
    res = 120. * (y - y2 * 0.015625 + y3 * 0.00137174211248285322);
    
    return (res);
}



//code from Rodrigo
//-------------------------------------------------------
//D4 and D5

double D4rodrigo (double eta, double Xn, double Xp) {
    double res;
    
    res = (Xn * F4m (eta) - Xp * F4p (eta)) / F4m (0.);
    
    return (res);
}

double D5rodrigo (double eta, double Xn, double Xp) {
    double res;
    
    res = (Xn * F5m (eta) + Xp * F5p (eta)) / F5m (0.);
    
    return (res);
}


//Fermi functions

double F4m (double x) {
    double res;

    res = 4. * 3. * 2. * (exp (-x) - exp (-2 * x) / 32. + exp (-3 * x) / 243.);

    return (res);
}

double F4p (double x) {
    double res;
    
    res = 7. * pow (M_PI, 4.) / 15. * x + 2 * pow (M_PI, 2) / 3. * pow (x, 3.)
        + pow (x, 5.) / 5. + F4m (x);

    return (res);
}

double F5m (double x) {
    double res;
    
    res = 5 * 4 * 3 * 2 * (exp (-x) - exp (-2 * x) / 64. + exp (-3 * x) / 729.);

    return (res);
}

double F5p (double x) {
    double res;
    
    res = 31 * pow (M_PI, 6.) / 126. + 7. * pow (M_PI, 4) / 6. * pow (x, 2.)
        + 5. * pow (M_PI, 2.) / 6. * pow (x, 4.) + pow (x, 6.) / 6. - F5m (x);
    
    return (res);
}
#endif
