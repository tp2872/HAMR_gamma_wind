//
//  nuclear.h
//  HARM2D
//
//  Created by Alexander Tchekhovskoy on 9/8/15.
//  Copyright (c) 2015 Home. All rights reserved.
//

#ifndef __HARM2D__nuclear__
#define __HARM2D__nuclear__

void nuc_evol(double pi[][N2M][N3M][NPR],double prh[][N2M][N3M][NPR], double pr[][N2M][N3M][NPR], double Dt, int i, int j, int k, int was_floor_activated);
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



#endif /* defined(__HARM2D__nuclear__) */
