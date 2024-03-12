//
//  nuclear.h
//  HARM2D
//
//  Created by Alexander Tchekhovskoy on 9/8/15.
//  Copyright (c) 2015 Home. All rights reserved.
//

#ifndef __HARM2D__nuclear__
#define __HARM2D__nuclear__

void nuc_evol (double pr[NPR], double Dt, int i, int j, int k, int was_floor_activated, int n);
#if DOHELM
void nuc_evol_helm (double pr[NPR], double Dt, int i, int j, int k, int was_floor_activated, int n);
#endif
double compute_rhounit();
double compute_Tunit();
double compute_dq_unit();
double F4(double eta);
double F5(double eta);
double F4neg(double eta);
double F5neg(double eta);
double D4(double eta, double Xn, double Xp);
double D5(double eta, double Xn, double Xp);
double compute_t_unit();
double compute_M_unit();
double compute_L_unit();
double compute_temperature(double rho, double p, double Ye);
double compute_degeneracy(double rho, double T, double Ye);
#endif /* defined(__HARM2D__nuclear__) */
