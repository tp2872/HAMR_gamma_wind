//
//  nuclear.c
//  HARM2D
//
//  Created by Alexander Tchekhovskoy on 9/8/15.
//  Copyright (c) 2015 Home. All rights reserved.
//

#include "decs.h"
#include "nuclear.h"

#if(DONUCLEAR)
double D4rodrigo(double eta, double Xn, double Xp);
double D5rodrigo(double eta, double Xn, double Xp);
double F4m(double x);
double F4p(double x);
double F5m(double x);
double F5p(double x);

void nuc_evol(double pi[][N2M][N3M][NPR],double prh[][N2M][N3M][NPR], double pr[][N2M][N3M][NPR], double Dt, int i, int j, int k, int was_floor_activated)
{
  //all sorts of nuclear physics
  double Xalpha, Xn, Xp, Xnp, Xfloor, Xamb, rho, rhofloor, rhotot, ug;
  double Xalphanew, dXalpha;
  double fac;
  double T, Ye;
  const double rho_unit = compute_rhounit();
  const double T_unit = compute_Tunit();
  const double t_unit = compute_t_unit();
  const double dq_unit = compute_dq_unit();
  const double M_unit = compute_M_unit();
  const double L_unit = compute_L_unit();
  struct of_geom geom;
  double ucon[NDIM];
  double etae, k_cgs, eV_cgs, MeV_cgs, T_MeV, rho_10, T_10, Xwb;
  double dqalpha;
  double G, Q, dG, dQ;
  double expmtaunu;
  double xxx;
  double r, th, phi, ufloor;
  //int ind, ind_max;            //for subcycling
  //double frac, mult, subcyc;
  int ind;                       //for implicit update of uu
  double Fnr, dFnr, a_nr, b_nr, c_nr, d_nr, e_nr;
 
  /////////
  //
  // Compute mass fractions: Xnp, Xalpha
  //
  
  //re-normalize densities
  if( pr[i][j][k][RHONP] < 0. ) pr[i][j][k][RHONP] = 0.;
  if( pr[i][j][k][RHOALPHA] < 0. ) pr[i][j][k][RHOALPHA] = 0.;
  if( pr[i][j][k][RHOFLOOR] < 0. ) pr[i][j][k][RHOFLOOR] = 0.;
  
  //total density
  rho = pr[i][j][k][RHO];
  ug = pr[i][j][k][UU];
  //floor+ambient density
  //rhofloor = pr[i][j][k][RHOFLOOR];

  //total physical density
  if(rho < 0. ) rho = 0.;
 
  fac = 1./(pr[i][j][k][RHONP]+pr[i][j][k][RHOALPHA]+pr[i][j][k][AMB]+SMALL);
 
  //rescale rho_alpha and rho_np to give the total density
  pr[i][j][k][RHOALPHA] *= fac;
  pr[i][j][k][AMB]      *= fac;
 
  //advected mass fractions 
  Xalpha = pr[i][j][k][RHOALPHA]; // / (rho+SMALL);
  Xamb = pr[i][j][k][AMB]; // / (rho+SMALL);

  //
  // End compute mass fractions
  //
  /////////
  
  //advected Ye
  Ye = pr[i][j][k][YE];

  //compute temperature accounting for a mixture of gas and neutrinos
  T = compute_temperature(rho, (gam-1)*ug, Ye);
  
  if( T < SMALL ) T = SMALL; //to avoid division by zero later on

  //compute degeneracy
  etae = compute_degeneracy(rho, T, Ye);
  
  //Xalpha + dXalpha = min(2Ye,2-2Ye) * (1-min(1,Xwb))
  k_cgs = 1.380658e-16; //erg/K
  eV_cgs = 1.6021772e-12; //erg/eV
  MeV_cgs = 1e-6 * k_cgs / eV_cgs;
  T_MeV = T*T_unit*MeV_cgs; //T in units of MeV
  rho_10 = rho * rho_unit * 1.e-10; //rho in units of 1e10 g/cm^3
  T_10 = T * T_unit * 1.e-10; //T in units of 1e10 K
  Xwb = 15.58*pow(T_MeV,1.125)*pow(rho_10,-0.75)*exp(-7.074/T_MeV);

  if (Xamb < 1 && T_10 > 0.5) {
    Xalphanew = MY_MIN(2.*Ye,2.*(1.-Ye)) * (1.-MY_MIN(1.,Xwb)) - Xamb;
    Xalphanew = MY_MAX(Xalphanew, 1.e-10);
    dXalpha = Xalphanew - Xalpha;
    //update Xalpha, Xn, and Xp
    Xalpha = Xalphanew;

    Xp = Ye - 0.5*Xalpha - Xamb;
    Xp = MY_MAX(Xp, 1.e-10);
  
    Xn = 1. - Xp - Xalpha - Xamb;
    Xn = MY_MAX(Xn, 1.e-10);

    //renormalize abundances
    fac = Xn + Xp + Xalpha + Xamb;
    Xn     /= fac;
    Xp     /= fac;
    Xalpha /= fac;
    Xamb   /= fac;

  } else {
    //abundances are frozen
    Xp = Ye - 0.5*Xalpha - Xamb;
    Xp = MY_MAX(Xp, 1.e-10);

    Xn = 1. - Xp - Xalpha - Xamb;
    Xn = MY_MAX(Xn, 1.e-10);

    dXalpha = 0.;

    //renormalize abundances
    fac = Xn + Xp + Xalpha + Xamb;
    Xn     /= fac;
    Xp     /= fac;
    Xalpha /= fac;
    Xamb   /= fac;

  }
  
  
  //compute u^t = ucon[0] = dt/dtau
  get_geometry(i,j,k,CENT,&geom) ;
  ucon_calc(pr[i][j][k], &geom, ucon) ;

  //get the true density floor
  get_phys_coord(i,j,k,&r,&th,&phi); 
  get_rho_u_floor(r, th, phi, &rhofloor, &ufloor);

  //update the mass fractions
  pr[i][j][k][RHONP] = (Xn+Xp); //*rho;
  pr[i][j][k][RHOALPHA] = Xalpha; //*rho;
  pr[i][j][k][AMB] = Xamb; //*rho;
  pr[i][j][k][RHOFLOOR] = rhofloor;
 
  //heat per unit mass converted to code units from cgs
  dqalpha = 6.8e18*dXalpha;  //heating in a time step [erg/g]
  dqalpha /= 9e20;  //erg/g = c^2
  if (T_10 <= 0.5 || rho <= 10.*rhofloor) {
    dqalpha = 0.;
  } 
 
  expmtaunu = exp(-rho*rho_unit/1.e11);
  G = 0.;
  if (rho > 10.*rhofloor) {
    G = 0.22 * pow(T_10,5.) * D4(etae,Xn,Xp) * expmtaunu; //[1/s]
  }
  //per unit mass dissipation rate, erg/g/s
  xxx = D5(etae,Xn,Xp);
//  if(i + mpi_startn[1] == 39/2 && j + mpi_startn[2] == 63/2 && k + mpi_startn[3] == 0) {
//      fprintf(stderr, "got here\n");
//  }
  Q = 0.;
  if (rho > 10.*rhofloor) {
    Q = -8.9e17 * pow(T_10,6.) * xxx * expmtaunu; //[erg/g/s]
  }
  //convert G and Q from cgs to code units and from rates to increments
  G /= 6.77e4;
  Q /= 6.08e25;  //[code units of energy per unit mass per unit time]
  //save G and Q to an array that's going to be written out to disk
  G_global[i][j][k] = G;
  Q_global[i][j][k] = Q;
  qalpha_global[i][j][k] = dqalpha*ucon[0]/Dt;
  //convert per unit mass heating rate into volumetric heating rate
  Q *= rho;
  dqalpha *= rho;
  //dx/dtau * dtau/dt = dx/dt, where ucon[0] = u^t = dt/dtau
  dG = G * Dt / ucon[0];
  dQ = Q * Dt / ucon[0];

  //apply nuclear heating directly to internal energy
  //pr[i][j][k][UU] += dqalpha + dQ;

  pr[i][j][k][YE] += dG;

  //implicit update of the internal energy when qalpha != 0.
  if (T_10 > 0.5) {

    Xalpha = Xalphanew -dXalpha;
    qalpha_global[i][j][k] = 0.;

    a_nr = (7.5657e-15*T_unit*T_unit/rho_unit)*(T_unit*T_unit/9e+20);
    b_nr = rho*6.8e18/9e20;
    c_nr = ug + dQ -b_nr*Xalpha; //RHS
    d_nr = MY_MIN(2.*Ye,2.*(1.-Ye));
    e_nr = 15.58*pow(rho_10,-0.75);


    //Newton-Raphson loop
    for (ind=0; ind<=50; ind++) {
      T     = compute_temperature(rho, (gam-1)*ug, Ye);
      T_MeV = T*T_unit*MeV_cgs;
      Xwb   = e_nr*pow(T_MeV,1.125)*exp(-7.074/T_MeV);
      Xalphanew = d_nr*(1.-MY_MIN(1.,Xwb)) - Xamb;
      Xalphanew = MY_MAX(Xalphanew, 1.e-10);

      Fnr = ug - b_nr*Xalphanew - c_nr; //function to zero
      if (Xwb >= 1.) {
        //Xalpha is constant and zero, update ug with cooling and return
        pr[i][j][k][UU] += dQ;
        break;
      }

      //derivative
      dFnr = 1. + b_nr*d_nr*(9./8. + 7.074/T_MeV)*Xwb/(ug + a_nr*pow(T,4.)/(gam-1.));

      //DEBUG:
      //if (i==15 && j==5 && mpi_rank==21)
      //  fprintf(stdout,"ug, Fnr/dFnr, rank, ind = %20.12e, %20.12e, %d, %d\n", ug, Fnr/dFnr,mpi_rank,ind);
      //fprintf(stdout,"i,j,k = %d, %d, %d\n", i,j,k);

      if (fabs(Fnr/dFnr) < 1e-8*ug) {

        //converged, update ug, qalpha, and abundances
        pr[i][j][k][UU] = ug;

        dqalpha = b_nr/rho*(Xalphanew-Xalpha);
        qalpha_global[i][j][k] = dqalpha*ucon[0]/Dt;

        Xalpha = Xalphanew;
        Xp = Ye - 0.5*Xalpha - Xamb;
        Xp = MY_MAX(Xp, 1.e-10);
        Xn = 1. - Xp - Xalpha - Xamb;
        Xn = MY_MAX(Xn, 1.e-10);

        fac = Xn + Xp + Xalpha + Xamb;
        Xn     /= fac;
        Xp     /= fac;
        Xalpha /= fac;
        Xamb   /= fac;
        pr[i][j][k][RHONP] = (Xn+Xp);
        pr[i][j][k][RHOALPHA] = Xalpha;
        pr[i][j][k][AMB] = Xamb;

        break;

      }

      ug = ug - Fnr/dFnr; //update ug

      if (ug < ufloor) {
        pr[i][j][k][UU] = ufloor;
        break;
      }

      if (ind==50) {
        fprintf(stderr, "N-R for ug did not converge\n" );
        //fprintf(stderr, "rho = %11.3e\n", rho*rho_unit);
        exit(2345);
      }

    }


    } else {
      //same as original, but dqalpha=0 for T_10 < 0.5
      pr[i][j][k][UU] += dqalpha + dQ;

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

double compute_rhounit()
{
  const double Mbh_cgs = 3.*1.99e33;
  const double mneutron_cgs = 1.6749286e-24;
  const double mneutron = mneutron_cgs/Mbh_cgs;
  const double G = 6.67259e-8;
  const double c = 2.99792458e10;
  const double hbar = 1.05457266e-27;
  const double rg_cgs = G*Mbh_cgs/(c*c);
  const double rho_unit = Mbh_cgs / (rg_cgs*rg_cgs*rg_cgs);
  return( rho_unit );
}

double compute_dq_unit()
{
  const double Mbh_cgs = 3.*1.99e33;
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
  return( dq_unit );
}

double compute_t_unit()
{
    const double Mbh_cgs = 3.*1.99e33;
    const double G = 6.67259e-8;
    const double c = 2.99792458e10;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    const double t_unit = rg_cgs/c;
    return( t_unit );
}

double compute_M_unit()
{
    const double Mbh_cgs = 3.*1.99e33;
    return( Mbh_cgs );
}


double compute_Tunit()
{
  const double c = 2.99792458e10;
  const double mneutron_cgs = 1.6749286e-24;
  const double k_cgs = 1.380658e-16; //erg/K
  const double T_unit = mneutron_cgs*c*c/k_cgs;
  return(T_unit);
}

double compute_L_unit()
{
    const double Mbh_cgs = 3 * 1.99e33;
    const double c = 2.99792458e10;
    const double G = 6.67259e-8;
    const double rg_cgs = G*Mbh_cgs/(c*c);
    return(rg_cgs);
}
//equation of state stuff: rho, p -> T
double compute_temperature(double rho, double p, double Ye)
{
  //COMPUTE THE TEMPERATURE (no nuclear recombination yet)
  
  const double Mbh_cgs = 3 * 1.99e33;
  const double mneutron_cgs = 1.6749286e-24;
  const double c = 2.99792458e10;
  const double G = 6.67259e-8;
  const double rg_cgs = G*Mbh_cgs/(c*c);
  const double hbar = 1.05457266e-27;
  const double lambdac_cgs = hbar / (mneutron_cgs*c);
  const double mneutron = mneutron_cgs/Mbh_cgs;
  const double lambdac = lambdac_cgs/rg_cgs;
  double atemp = (M_PI*M_PI/45.) * (1./p) * mneutron * pow(lambdac,-3.);
  double btemp = (1. + Ye)*(rho/p);
  double Fzero, dFzero;
  
  //initial guess assuming radiation-pressure-dominance
  double x = pow(atemp,-0.25);
  
  double prec_newton = 1e-6;
  int ind;
  const int max_iter = 50;
  
  for( ind=1; ind < max_iter; ind++ ) {
    Fzero  =    atemp*pow(x,4.) + btemp*x - 1.;
    dFzero = 4.*atemp*pow(x,3.) + btemp;
    
    x = x - Fzero/dFzero;
    if (fabs(Fzero/dFzero) < prec_newton*x) break;
  }
  if (max_iter == ind) {
    fprintf(stderr, "Cool: N-R for temperature did not converge\n" );
    fprintf(stderr, "rho = %11.3e\n", rho*Mbh_cgs/pow(rg_cgs,3.));
    fprintf(stderr, "p   = %11.3e\n", p*Mbh_cgs*c*c/pow(rg_cgs,3.));
    fprintf(stderr, "Ye  = %11.3e\n", Ye);
    exit(2345);
  }
  return(x);
}

//accepts temperature in code units T = k Tcgs / mneutron_cgs c^2
double compute_degeneracy(double rho, double T, double Ye)
{
  //Compute degeneracy parameter
  const double Mbh_cgs = 3.*1.99e33;
  const double mneutron_cgs = 1.6749286e-24;
  const double mneutron = mneutron_cgs/Mbh_cgs;
  const double G = 6.67259e-8;
  const double c = 2.99792458e10;
  const double hbar = 1.05457266e-27;
  const double rg_cgs = G*Mbh_cgs/(c*c);
  const double rho_unit = Mbh_cgs / (rg_cgs*rg_cgs*rg_cgs);
  const double k_cgs = 1.380658e-16;
  double ANN, BOB;
  const double CHARLIE = (M_PI*M_PI*M_PI)*(M_PI*M_PI*M_PI)/27.;
  double pf_cgs = hbar * pow(3.*M_PI*M_PI*Ye*rho*rho_unit/mneutron_cgs,1./3.);
  double T_cgs = T*mneutron_cgs*c*c/k_cgs;
  double pfcokT = pf_cgs * c / (k_cgs * T_cgs);
  double pfcokT3 = (pfcokT*pfcokT)*pfcokT;
  double pfcokT6 = pfcokT3*pfcokT3;
  double etae;
  
  if (T_cgs > 1.1604519308e+10) {
    ANN = 0.5*pfcokT3;
    BOB = sqrt( 0.25*pfcokT6 + CHARLIE );
    etae = pow(ANN+BOB,1./3.) - pow(BOB-ANN,1./3.);
  }
  else {
    etae = 0.;
  }
  return(etae);
}

double D4sasha(double eta, double Xn, double Xp);
double D5sasha(double eta, double Xn, double Xp);

double D4(double eta, double Xn, double Xp)
{
    double a, b, diff;
//    a = D4sasha(eta,Xn,Xp);
    b = D4rodrigo(eta,Xn,Xp);
//    diff = 2.*fabs(a-b)/(fabs(a)+fabs(b)+SMALL);
//    if(diff>1e-12) {
//        fprintf(stderr, "D4diff = %g; s = %21.15g, r = %21.15g, eta = %g, Xn = %g, Xp = %g\n", diff, a, b, eta, Xn, Xp);
//    }
    return(b);
}

double D5(double eta, double Xn, double Xp)
{
    double a, b, diff;
//    a = D5sasha(eta,Xn,Xp);
    b = D5rodrigo(eta,Xn,Xp);
//    diff = 2.*fabs(a-b)/(fabs(a)+fabs(b)+SMALL);
//    if(diff>1e-12) {
//        fprintf(stderr, "D5diff = %g; s = %21.15g, r = %21.15g, eta = %g, Xn = %g, Xp = %g\n", diff, a, b, eta, Xn, Xp);
//        fprintf(stderr, "s: F5m = %g; F5p = %21.15g, r: F5m = %g, F5p = %g\n",
//                F5neg(-eta), F5(eta), F5m(eta), F5p(eta));
//    }
    return(b);
}
double D4sasha(double eta, double Xn, double Xp)
{
    double res;
    res = (Xn*F4(-eta)-Xp*F4(eta))/F4(0);
    return( res );
}

double D5sasha(double eta, double Xn, double Xp)
{
    double res;
    res = (Xn*F5(-eta)+Xp*F5(eta))/F5(0);
    return( res );
}

double F4(double eta)
{
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

double F4neg(double eta)
{
    double res;
    double y, y2, y3;
    
    y  = exp(eta);
    y2 = y*y;
    y3 = y2*y;
    
    res = 24. * (y - y2*0.03125 + y3*0.00411522633744855967 );
    
    return( res );
}

double F5neg(double eta)
{
    double res;
    double y, y2, y3;
    
    y  = exp(eta);
    y2 = y*y;
    y3 = y2*y;
    
    res = 120. * (y - y2*0.015625 + y3*0.00137174211248285322 );
    
    return( res );
}


//code from Rodrigo
//-------------------------------------------------------
//D4 and D5


double D4rodrigo(double eta, double Xn, double Xp)
{
    double res;
    res = (Xn*F4m(eta) - Xp*F4p(eta))/F4m(0.);
    return(res);
}
double D5rodrigo(double eta, double Xn, double Xp)
{
    double res;
    res = (Xn*F5m(eta) + Xp*F5p(eta))/F5m(0.);
    return(res);
}

//Fermi functions
double F4m(double x)
{
    double res;

    res = 4.*3.*2.*(exp(-x)-exp(-2*x)/32. + exp(-3*x)/243.);

    return(res);
}
//end function F4m

double F4p(double x)
{
    double res;
    res = 7.*pow(M_PI,4.)/15.*x + 2*pow(M_PI,2)/3.*pow(x,3.)
        + pow(x,5.)/5. + F4m(x);

    return(res);
}
//end function F4p


double F5m(double x)
{
    double res;
    res = 5*4*3*2*(exp(-x) -exp(-2*x)/64. + exp(-3*x)/729.);

    return(res);
}
//end function F5m

double F5p(double x)
{
    double res;
    
    res = 31*pow(M_PI,6.)/126. + 7.*pow(M_PI,4)/6.*pow(x,2.)
        + 5.*pow(M_PI,2.)/6.*pow(x,4.) + pow(x,6.)/6. -F5m(x);
    return(res);
}
//end function F5p


#endif











