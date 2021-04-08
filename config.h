/*************************************************************************
Physical Parameters section
*************************************************************************/
/*Select Desired problem, see init.c for implementation*/
#define MONOPOLE_PROBLEM_1D 1
#define MONOPOLE_PROBLEM_2D 2
#define BZ_MONOPOLE_2D 3
#define TORUS_PROBLEM 4
#define DISRUPTION_PROBLEM 5
#define BONDI_PROBLEM_1D 6
#define BONDI_PROBLEM_2D 7
#define TORUS_PROBLEM_GRB 8
#define THIN_PROBLEM 9
#define SOUND_WAVE 10
#define ENT_WAVE 11
#define TRUNC_PROBLEM 12
#define POSTMERGER_PROBLEM 13

#define WHICHPROBLEM TORUS_PROBLEM

/*Set Cartesian grid for test problems*/
#define CARTESIAN (0)

/*Enable special refinement criterion for large scale jet simulations*/
#define REFINE_JET (0)

/*Gibwa's refinement criterion*/
#define REFINE_GIBWA (0)

/*Select adiabatic index and BH spin*/
#define GAMMA	(4./3.)
#define BH_SPIN (0.9375)

/*Wheter or not to tilt the disk*/
#define TILTED (0)
#define TILT_ANGLE (45.0)

/*Wheter to activate an untilted elliptical disk*/
#define ELLIPTICAL (0)
#define ELLIPTICAL2 (0)

/*Wheter to cool the disk to predifined thickness H_OVER_R. Not implemented in CPU version*/
#define COOL_DISK (0)
#define H_OVER_R (0.05)

/*Wheter or not to use the full dispersion relation. Only slows down simulation and does not really increase accuracy. Do not use, not implemented anymore*/
#define FULL_DISP (0)

/* Whether Helmholtz EOS is used; defined before the FIXUP parameters to set floors for torus problem */
#define DOHELM (0)
#define KTOT_FACTOR (1e-10)
#define EOS_BISECTION_THRESHOLD (1e6)

/** FIXUP PARAMETERS, magnitudes of rho and u, respectively, in the floor : **/
#if( (WHICHPROBLEM == POSTMERGER_PROBLEM))
#if (DOHELM)
// Danat: otherwise EOS fails, since the densities are too low outside the torus
    #define RHOMIN (1.e-14)     
    #define UUMIN (1.e-16)      
    #define RHOMINLIMIT (1.e-20)
    #define UUMINLIMIT (1.e-20) 
#else
    #define RHOMIN      (1.e-26)
    #define UUMIN       (1.e-27)
    #define RHOMINLIMIT (1.e-40)
    #define UUMINLIMIT  (1.e-40)
#endif
#elif ((DOHELM) && (WHICHPROBLEM == TORUS_PROBLEM))
    #define RHOMIN    (1.e-14)
    #define UUMIN    (1.e-16)
    #define RHOMINLIMIT (1.e-30)
    #define UUMINLIMIT  (1.e-30)
#else
    #define RHOMIN	(1.e-7)
    #define UUMIN	(1.e-9)
    #define RHOMINLIMIT (1.e-20)
    #define UUMINLIMIT  (1.e-20)
#endif
#define POWRHO (2.0)
#define FLOORFACTOR (1.0)
#define BSQORHOMAX (15.*FLOORFACTOR)
#define BSQOUMAX (750.*FLOORFACTOR)
#define UORHOMAX (150.*FLOORFACTOR)

/* Max. value of gamma, the lorentz factor */
#define GAMMAMAX (80.)
#define GAMMAMAX_RAD (50.000625)

/*Max value of electron temperature in Kelvin*/
#define TMAX (1.e15)

/*Runtime in hours*/
#define RUNTIME (24.0)

/*************************************************************************
Numerical Parameters section
*************************************************************************/
/*Whether or not to use the 3D version of the code*/
#define ThreeD (1)

/*Set execution mode. Note that GPU needs double precision support. Enable CPU_OPENMP to run on CPU. Do not use GPU_DEBUG*/
#define GPU_ENABLED 1
#define GPU_DEBUG 0
#define CPU_OPENMP 0
#define TIMER 1

/*Enable AMD for FMA instructions, works also good with NVIDIA now!*/
#define AMD (0)

/*Enable if running on the new VOLTA GPUs*/
#define V100 (1)

/*Use NVIDIA GPU_DIRECT. Check availability on cluster and enable it in slurm job script, for mpich set MPICH_RDMA_ENABLED_CUDA=1*/
#define GPU_DIRECT 1

/*Maximum tag number for MPI messages so not to overflow*/
#define MPI_TAG_MAX 1264576

/*Enable parallel I/0*/
#define PARALLEL_IO (0)

/*Determine if you want to explicitely copy the B fields from block to block. Good to use when working on AMR, since a good implementation gives divB=0*/
#define COPY_BFIELD 1

/*Maximum number of blocks per node and hten umber of memory places(should be equal)*/
#define MAX_BLOCKS (40)
#define NB_LOCAL (1200)

/*Define number of blocks for the first AMR level in all three dimensions*/
#define NB_1 6
#define NB_2 2
#define NB_3 1

/*Set block size in each dimension*/
#define BS_1 302
#define BS_2 512
#define BS_3 1

/*Set the maximum number of refinement levels*/
#define N_LEVELS_3D 1

/*Set in which dimensions to refine for AMR. Do not change, deprecated!*/
#if(BS_1==1)
#define REF_1 0
#else
#define REF_1 1
#endif
#if(BS_2==1)
#define REF_2 0
#else
#define REF_2 1
#endif
#if(BS_3==1)
#define REF_3 0
#else
#define REF_3 1
#endif

/*Number of GPUs per MPI rank*/
#define N_GPU 1

/*If you want to call multiple blocks from multiple threads. Will not *allways* improve performance and SLOWS down performance of workstation, so not recommended for non-cluster use!*/
#define GPU_OPENMP 0

/*Derefines the pole in the third dimension. Make sure REF_3==1 and NB_2=6,12,24,48 and NB_1=4 and NB_3>=2*/
#define DEREFINE_POLE (0)

/*Number of internal derefinement levels*/
#define N_LEVELS_1D_INT (0)

/*Enable very fast hierarchical timestepping routine in combination with DEREFINE_POLE and REF_1=0, REF_2=0, REF_3=1. Do not use! Deprecated: With new load balancing and AMR there is no speedup*/
#define TIMESTEP_JET 0

//Use Z-order at 0-level for load balancing
#define Z_ORDER 1

/*Set the maximum weight for load balancing of a heavy block around the pole*/
#define MAX_WEIGHT (1)

/*Set maximum timelevel for AMR (ie 1,2,4,8 etc). This determines how often the timestep is changed so setting it to an absurd high value may cause code crashes
If a very high value is needed, lowerin Courant factor may increase stability*/
#define AMR_MAXTIMELEVEL 8

/*The minimum timeinterval at which refinement takes place, TREF can't go below it*/
#define AMR_SWITCHTIMELEVEL 8

/*Minimum number of step times AMR_SWITCHTIMELEVEL for checkppointing to proceed*/
#define DUMPFACTOR (300)

/*Use prestepping for load balancing with HTS*/
#define PRESTEP 0

/*Use second order timestepping at LAS boundaries, not possible in combination with PRESTEP*/
#define PRESTEP2 0

/*Used for loading in old data files. Do not touch!*/
#define REVERSE_ORDERING 0

//The time between refinement (AMR) steps
#define TREF 500.

/*Select the courant factor for the timestep*/
#define COUR (0.8)

/*Evolve entropy for more stability*/
#define DO_FONT_FIX (1) //Use redundant inversion scheme for more stability
#define DOKTOT 1  //Evolve entropy to do the above even more accurately
#define FULL_ENTROPY (0) //Evolve the full entropy equation S=1/(gamma-1)*log(P/rho^gamma) instead of the entropy tracer K=p/rho^gamma

/*Enable/disable PPM spatial reconstruction. Never enable both*/
#define PPM (1)
#define PPM_FLATTENER (0)

/*Enable/disable van Leer spatial reconstruction. Never enable both*/
#define LEER (0) //Not working

/*Wheter to set floors in ZAMO frame*/
#define ZAMO_FLOOR (0)

/*Wheter to set floors in drift frame*/
#define DRIFT_FLOOR (1)

/*Whether or not to allow inflow for fluxes (see fix_flux())*/
#define INFLOW 0

/*Enable or disable the HLLC solver.*/
#define HLLC (0)

/*Enable or disable the HLLD solver. Does not work yet!*/
#define HLLD (0)

/*Whether or not to use a stagger magnetic field*/
#define STAGGERED (1)

/*Whether or not to use a stagger electric field*/
#define STAGGERED_E (0)

/*Wheter or not to use a non symmetric metric for tilted disk. Not fully implemented in this version!*/
#define NSY (0)

/*Use transmissive boundary condition at pole*/
#define TRANS_BOUND (1*((BS_3*NB_3)>1) && !CARTESIAN)

/* how many cells near the poles to stabilize, choose 0 for no stabilization */
#define POLEFIX 2

/*Set periodic boundary conditions only in the third dimension is supported*/
#define PERIODIC1 CARTESIAN
#define PERIODIC2 CARTESIAN
#if (BS_3*NB_3==1)
#define PERIODIC3 0
#else
#define PERIODIC3 1
#endif

/* A numerical convenience to represent a small non-zero quantity compared to unity:*/
#define SMALL	(1.e-20)

/* maximum fractional increase in timestep per timestep */
#define SAFE	(1.3)

#define COORDSINGFIX 1
// whether to move polar axis to a bit larger theta
// theta value where singularity is displaced to
#define SINGSMALL (1.E-20)

/*Define local work size for GPU, for NVIDIA Kepler,Pascal, Volta and AMD GCN chose 64*/
#define LOCAL_WORK_SIZE 64

/*Set grid parameters X1*/
#define RADEXP 1.0
#define RTRANS 5000000.
#define RB  0.

/*Set grid parameters X2*/
//Big torus, very strongly collimating
//#define BRAVO (0.6)
//#define TANGO (1.0)
//#define CHARLIE (0.8)
//#define DELTA (3.0)

//Uniform Grid
#define BRAVO (0.0)
#define TANGO (1.0)
#define CHARLIE (0.0)
#define DELTA (3.0)

/*Wheter to cylindrify coordinates to increase GLOBAL timestep. Not usefull with internal derefinement, may become deprecated!*/
#define DOCYLINDRIFYCOORDS (0)

/*Put out files which Ziri can Ray-Trace. Not fully implemented yet*/
#define ZIRI_DUMP 0

/*Whether to output a reduced resolution file*/
#define DUMP_SMALL (0)
#define REDUCE_FACTOR1 (4)
#define REDUCE_FACTOR2 (4)
#define REDUCE_FACTOR3 (4)

/*Whether to dump diag file*/
#define DUMP_DIAG (0)

/*Enable MPI; Old remnant do not touch!*/
#define MPI_enable 1

/*Enable Radiation*/
#define RAD_M1 (1)

/*Enable advenced Roseland and energy opacities*/
#define OP_EXTRA (0)

/*Enable photon number evolution*/
#define P_NUM (0)

/*Enable 2-temperature evolution*/
#define TWO_T (0)

/*Wheter to use fixed or variable gamma*/
#define FIXEDGAMMA (1)

/*Electron gamma-->electrons are most of the time relativistic, so 4/3 is appropriate*/
#define GAMMAE (4./3.)

/*Enable or disable library with Bessel functions*/
#define GSL_ENABLED (0)

/*Enable Resistivity*/
#define RESISTIVE (0)

/*Set resistivity coefficient*/
#define ETA (0.0)

/*Enable IMEX*/
#define DO_IMEX (0)

/* use local lax-friedrichs or HLL flux:  these are relative weights on each numerical flux */
#if(RESISTIVE || RAD_M1)
#define HLLF  (0)
#define LAXF  (1)
#else
#define HLLF  (1)
#define LAXF  (0)
#endif

//Abundace constants
#define Z_AB (0.02)
#define Y_AB (0.28)
#define X_AB (0.70)

// CGS constants needed for radiation
#define ARAD (7.5657e-15) /*Radiation density constant*/
#define MH_CGS (1.673534e-24) /*Mass hydrogen molecule*/
#define ME_CGS (9.1094e-28) /*Mass hydrogen molecule*/
#define MMW (1.69) /*Mean molecular weight*/
#define BOLTZ_CGS (1.3806504e-16) /*Boltzmanns constant*/
#define THOMSON_CGS (6.652e-25) /*Thomson cross section*/
#define PLANCK_CGS (6.6260755e-27) /*Planck's constant*/
#define STEFAN_CGS (5.67051e-5) /*Stefan-Boltzmann constant*/
#define FINE_CGS (7.29735308e-3)
#define ERM_CGS (9.10938215e-28) /*Electron rest mass*/
#define E_CGS (4.80320427e-10) /*Elementary charge*/
#define C_CGS (2.99792458e10) /*Speed of light*/
#define M_SGRA_SOLAR (1.0e1) /* Solar masses */
#define M_SOLAR_CGS (1.998e33) /* Solar mass */
#define G_CGS (6.67259e-8) /* Gravitational constant */
#define MU_I (4.0/(4.0*X_AB+Y_AB))
#define MU_E (2.0/(1.0+X_AB))
#define MU_G (4.0/(6*X_AB+Y_AB+2.0))
#define BASIC (0)
#define TYPE2 (1)
#define IONS (0)
#define ELECTRONS (1)

// Scaling from code units to cgs units
#define R_G_CGS (M_SGRA_SOLAR * M_SOLAR_CGS * G_CGS / (C_CGS * C_CGS)) /*Gravitational radius*/
#define R_GOC_CGS (R_G_CGS / C_CGS) /*Light-crossing time*/
#define MASS_DENSITY_SCALE (3.1)
#define ENERGY_DENSITY_SCALE (MASS_DENSITY_SCALE * C_CGS * C_CGS)
#define MAGNETIC_DENSITY_SCALE (sqrt(MASS_DENSITYSCALE) * C_CGS)
#define PRESSURE_SCALE (MASS_DENSITY_SCALE * C_CGS * C_CGS)

//IMEX constant
#define Y_IMEX (0.29289321881)

/*************************************************************************
MNEMONICS SECTION
*************************************************************************/
/* mnemonics for primitive vars; conserved vars */
#define RHO	(0)	
#define UU	(1)
#define U1	(2)
#define U2	(3)
#define U3	(4)
#define B1	(5)
#define B2	(6)
#define B3	(7)
#define KTOT (8)
#define UU_RAD	(8+DOKTOT)
#define U1_RAD	(8+DOKTOT+1)
#define U2_RAD	(8+DOKTOT+2)
#define U3_RAD	(8+DOKTOT+3)
#define E1 (8+DOKTOT+RAD_M1*4)
#define E2 (8+DOKTOT+RAD_M1*4+1)
#define E3 (8+DOKTOT+RAD_M1*4+2)
#define ENTRE (8+DOKTOT+RAD_M1*4+RESISTIVE*3)
#define ENTRI (8+DOKTOT+RAD_M1*4+RESISTIVE*3+1)
#define PHOTON (8+DOKTOT+RAD_M1*4+RESISTIVE*3+TWO_T*2)

/* mnemonics for centering of grid functions */
#define LEFT (0)
#define RIGHT (1)
#define FACE1	(0)	
#define FACE2	(1)
#define CORN	(2)
#define CENT	(3)
#define FACE3	(4)

//For variable inversions
#define UTCON1 	2
#define UTCON2 	3
#define UTCON3 	4
#define BCON1	5
#define BCON2	6
#define BCON3	7

//For variable inversions
#define QCOV0	1
#define QCOV1	2
#define QCOV2	3
#define QCOV3	4

/* mnemonics for slope limiter */
#define MC	(0)
#define VANL	(1)
#define MINM	(2)

/* mnemonics for diagnostic calls */
#define INIT_OUT	    (0)
#define DUMP_OUT	    (1)
#define IMAGE_OUT	    (2)
#define LOG_OUT		    (3)
#define FINAL_OUT	    (4)
#define DUMP_OUT_REDUCED	(5)

/* failure modes */
#define FAIL_UTOPRIM        (1)
#define FAIL_VCHAR_DISCR    (2)
#define FAIL_COEFF_NEG	    (3)
#define FAIL_COEFF_SUP	    (4)
#define FAIL_GAMMA          (5)
#define FAIL_METRIC         (6)

/*For Windows users*/
#ifndef M_PI 
#define M_PI 3.14159265358979323846264338327950288 
#define M_PI_2 (M_PI*0.5)
#endif 

/*Mnemonics for AMR parameters*/
#define NV 183
#define AMR_ACTIVE 0
#define AMR_LEVEL 1
#define AMR_REFINED 2
#define AMR_COORD1 3
#define AMR_COORD2 4
#define AMR_COORD3 5
#define AMR_PARENT 6
#define AMR_CHILD1 7
#define AMR_CHILD2 8
#define AMR_CHILD3 9
#define AMR_CHILD4 10
#define AMR_CHILD5 11
#define AMR_CHILD6 12
#define AMR_CHILD7 13
#define AMR_CHILD8 14
#define AMR_NBR1 15
#define AMR_NBR2 16
#define AMR_NBR3 17
#define AMR_NBR4 18
#define AMR_NBR5 19
#define AMR_NBR6 20
#define AMR_CORN1 21
#define AMR_CORN2 22
#define AMR_CORN3 23
#define AMR_CORN4 24
#define AMR_CORN5 25
#define AMR_CORN6 26
#define AMR_CORN7 27
#define AMR_CORN8 28
#define AMR_CORN9 29
#define AMR_CORN10 30
#define AMR_CORN11 31
#define AMR_CORN12 32
#define AMR_NODE 33
#define AMR_POLE 34
#define AMR_NUMBER 35
#define AMR_TIMELEVEL 36
#define AMR_TAG 37
#define AMR_CORN1D 38
#define AMR_CORN2D 39
#define AMR_CORN3D 40
#define AMR_CORN4D 41
#define AMR_CORN5D 42
#define AMR_CORN6D 43
#define AMR_CORN7D 44
#define AMR_CORN8D 45
#define AMR_CORN9D 46
#define AMR_CORN10D 47
#define AMR_CORN11D 48
#define AMR_CORN12D 49
#define AMR_CORN1D_1 50
#define AMR_CORN2D_1 51
#define AMR_CORN3D_1 52
#define AMR_CORN4D_1 53
#define AMR_CORN5D_1 54
#define AMR_CORN6D_1 55
#define AMR_CORN7D_1 56
#define AMR_CORN8D_1 57
#define AMR_CORN9D_1 58
#define AMR_CORN10D_1 59
#define AMR_CORN11D_1 60
#define AMR_CORN12D_1 61
#define AMR_CORN1D_2 62
#define AMR_CORN2D_2 63
#define AMR_CORN3D_2 64
#define AMR_CORN4D_2 65
#define AMR_CORN5D_2 66
#define AMR_CORN6D_2 67
#define AMR_CORN7D_2 68
#define AMR_CORN8D_2 69
#define AMR_CORN9D_2 70
#define AMR_CORN10D_2 71
#define AMR_CORN11D_2 72
#define AMR_CORN12D_2 73
#define RM_ORDER 74
#define GDUMP_WRITTEN 75
#define AMR_PRESTEP 76
#define AMR_GPU 77
#define AMR_NSTEP 78
#define AMR_IPROBE1 79
#define AMR_IPROBE1_1 80
#define AMR_IPROBE1_2 81
#define AMR_IPROBE1_3 82
#define AMR_IPROBE1_4 84
#define AMR_IPROBE2 85
#define AMR_IPROBE2_1 86
#define AMR_IPROBE2_2 87
#define AMR_IPROBE2_3 88
#define AMR_IPROBE2_4 89
#define AMR_IPROBE3 90
#define AMR_IPROBE3_1 91
#define AMR_IPROBE3_2 92
#define AMR_IPROBE3_3 93
#define AMR_IPROBE3_4 94
#define AMR_IPROBE4 95
#define AMR_IPROBE4_1 96
#define AMR_IPROBE4_2 97
#define AMR_IPROBE4_3 98
#define AMR_IPROBE4_4 99
#define AMR_IPROBE5 100
#define AMR_IPROBE5_1 101
#define AMR_IPROBE5_2 102
#define AMR_IPROBE5_3 103
#define AMR_IPROBE5_4 104
#define AMR_IPROBE6 105
#define AMR_IPROBE6_1 106
#define AMR_IPROBE6_2 107
#define AMR_IPROBE6_3 108
#define AMR_IPROBE6_4 109
#define AMR_LEVEL1 110
#define AMR_LEVEL2 111
#define AMR_LEVEL3 112
#define AMR_NBR1_3 113
#define AMR_NBR1_4 114
#define AMR_NBR1_7 115
#define AMR_NBR1_8 116
#define AMR_NBR2_1 117
#define AMR_NBR2_2 118
#define AMR_NBR2_3 119
#define AMR_NBR2_4 120
#define AMR_NBR3_1 121
#define AMR_NBR3_2 122
#define AMR_NBR3_5 123
#define AMR_NBR3_6 124
#define AMR_NBR4_5 125
#define AMR_NBR4_6 126
#define AMR_NBR4_7 127
#define AMR_NBR4_8 128
#define AMR_NBR5_1 129
#define AMR_NBR5_3 130
#define AMR_NBR5_5 131
#define AMR_NBR5_7 132
#define AMR_NBR6_2 133
#define AMR_NBR6_4 134
#define AMR_NBR6_6 135
#define AMR_NBR6_8 136
#define AMR_CORN1_1 137
#define AMR_CORN1_2 138
#define AMR_CORN2_1 139
#define AMR_CORN2_2 140
#define AMR_CORN3_1 141
#define AMR_CORN3_2 142
#define AMR_CORN4_1 143
#define AMR_CORN4_2 144
#define AMR_CORN5_1 145
#define AMR_CORN5_2 146
#define AMR_CORN6_1 147
#define AMR_CORN6_2 148
#define AMR_CORN7_1 149
#define AMR_CORN7_2 150
#define AMR_CORN8_1 151
#define AMR_CORN8_2 152
#define AMR_CORN9_1 153
#define AMR_CORN9_2 154
#define AMR_CORN10_1 155
#define AMR_CORN10_2 156
#define AMR_CORN11_1 157
#define AMR_CORN11_2 158
#define AMR_CORN12_1 159
#define AMR_CORN12_2 160
#define AMR_NBR1P 161
#define AMR_NBR2P 162
#define AMR_NBR3P 163
#define AMR_NBR4P 164
#define AMR_NBR5P 165
#define AMR_NBR6P 166
#define AMR_CORN1P 167
#define AMR_CORN2P 168
#define AMR_CORN3P 169
#define AMR_CORN4P 170
#define AMR_CORN5P 171
#define AMR_CORN6P 172
#define AMR_CORN7P 173
#define AMR_CORN8P 174
#define AMR_CORN9P 175
#define AMR_CORN10P 176
#define AMR_CORN11P 177
#define AMR_CORN12P 178
#define AMR_TAG1 179
#define AMR_TAG3 180
#define AMR_WEIGHT 181
#define GDUMP_WRITTEN_REDUCED 182


/*************************************************************************
Variable Inversion Section
*************************************************************************/
#define G_ISOTHERMAL (1.)

/* use K(s)=K(r)=const. (G_ATM = GAMMA) of time or  T = T(r) = const. of time (G_ATM = 1.) */
#define USE_ISENTROPIC 1

#if( USE_ISENTROPIC ) 
#define G_ATM GAMMA
#else
#define G_ATM G_ISOTHERMAL
#endif

//Use Newman&Hamhin inversion
#define NEWMAN (0)

#define MAX_NEWT_ITER 30     /* Max. # of Newton-Raphson iterations for find_root_2D(); */
#define NEWT_TOL   1.0e-10    /* Min. of tolerance allowed for Newton-Raphson iterations */
#define MIN_NEWT_TOL  1.0e-10    /* Max. of tolerance allowed for Newton-Raphson iterations */
#define EXTRA_NEWT_ITER 2
#define NEWT_TOL2     1.0e-15      /* TOL of new 1D^*_{v^2} gnr2 method */
#define MIN_NEWT_TOL2 1.0e-10  /* TOL of new 1D^*_{v^2} gnr2 method */
#define W_TOO_BIG	1.e20	/* \gamma^2 (\rho_0 + u + p) is assumedto always be smaller than this.  Thisis used to detect solver failures */
#define UTSQ_TOO_BIG	1.e20    /* \tilde{u}^2 is assumed to be smallerthan this.  Used to detect solverfailures */

#define FAIL_VAL  1.e30    /* Generic value to which we set variables when a problem arises */
#define NUMEPSILON (2.2204460492503131e-16)

/*Set dimensions for Utoprim routines*/
#define NEWT_DIM_2 2
#define NEWT_DIM_1 1

/*************************************************************************
Section with EOS constants
*************************************************************************/
#define EOSIMAX (541)   
#define EOSJMAX (201)   
// Log10 of EOS quantity limits
#define eos_tlo (3.0)   
#define eos_dlo (-12.0) 
#define eos_thi (13.0)
#define eos_dhi (15.0)
// EOS quantity limits
#define eos_temp_low (1e3)
#define eos_temp_up (1e13)
#define eos_dens_low (1e-12)
#define eos_dens_up (1e15)

#define MAXLEN (1024)

// tolerances 
#define EOS_TEMP_TOL (1.e-10)
#define EOS_TOL (1.e-10)
#define EOS_ITERATIONS (50)

// becomes true if variables for Aprox13t network are set
#define bAprox13t (0)

// Use linear interpolation of the EOS table
#define EOS_LINEAR (1)

// if you set eos_coulombAbort to non-zero, set EOS_COULOMB_CORR to 1
// otherwise, set EOS_COULOMB_CORR to 0
#define eos_coulombMult (0.0)
#define EOS_COULOMB_CORR (0)

#define eos_coulombAbort (1)

// from eos_helmConstData
#define avo (6.0221367e23)
#define kerg (1.380658e-16)
#define kev (8.617385e-5)
#define amu (1.6605402e-24)
#define avoinv (1.0e0 / avo)
#define kergavo (kerg * avo)
#define c_light (2.99792458e10)
#define h_planck (6.6260755e-27)
#define hbar_planck (1.05457266e-27)
#define ssol (5.67051e-5)
#define asol (4.0e0 * ssol / c_light)
#define asoli3 (asol / 3.0e0)
#define asoli3_inv (3.0e0 / asol)
#define sioncon ((2.0e0 * M_PI * amu * kerg) / (h_planck * h_planck))

#if (WHICHPROBLEM == POSTMERGER_PROBLEM)
#define Mbh_cgs (8.07 * 1.99e33)
#else
#define Mbh_cgs (3 * 1.99e33)
#endif 

#define G_cgs (6.67259e-8)

#define third (1.0e0/3.0e0)
#define forth (4.0e0/3.0e0)
#define eos_qe (4.8032068e-10)
#define esqu (eos_qe * eos_qe)

// conversion factors for EOS
// DANAT: finish!
#define conv_T_CODE2CGS (1.0)
#define conv_dens_CODE2CGS MASS_DENSITY_SCALE //(c_light * c_light * c_light * c_light * c_light * c_light / (G_cgs * G_cgs * G_cgs * Mbh_cgs * Mbh_cgs)) // = c_light^6 / G_cgs^3 / M_bh^2
#define conv_dens_CGS2CODE (1.0 / MASS_DENSITY_SCALE) //(G_cgs * G_cgs * G_cgs * Mbh_cgs * Mbh_cgs) / (c_light * c_light * c_light * c_light * c_light * c_light)
#define conv_pres_CODE2CGS PRESSURE_SCALE //((c_light * c_light * c_light * c_light * c_light * c_light * c_light * c_light) / (G_cgs * G_cgs * G_cgs * Mbh_cgs * Mbh_cgs))
#define conv_pres_CGS2CODE (1.0 / PRESSURE_SCALE) //(G_cgs * G_cgs * G_cgs * Mbh_cgs * Mbh_cgs / (c_light * c_light * c_light * c_light * c_light * c_light * c_light * c_light)) // = G_cgs^3 * M_bh^2 /c_light^8
#define conv_ener_CODE2CGS (c_light * c_light)
#define conv_ener_CGS2CODE (1.0 / (c_light * c_light)) // = 1 / c_light^2
#define conv_entr_CGS2CODE (1.0 / kergavo)

//For the uniform background coulomb correction
#define eos_a1 (-0.898004e0)
#define eos_b1 (0.96786e0)
#define eos_c1 (0.220703e0)
#define d1cc (-0.86097e0)
#define e1cc (2.5269e0)
#define eos_a2 (0.29561e0)
#define eos_b2 (1.9885e0)
#define eos_c2 (0.288675e0)
#define third (1.0e0/3.0e0)
#define forth (4.0e0/3.0e0)

//For the nuclear physics: alpha particles
#define Qalpha (4.5334641147464686e-5)
#define Qa (28.3 * 1.60217733e-6)

// ***********Beginning of statement function declarations **********
// quintic hermite polynomial statement functions
// psi0 and its derivatives
#define psi0(zFunc) (zFunc*zFunc*zFunc * ( zFunc * (-6.0e0*zFunc + 15.0e0) -10.0e0) + 1.0e0)
#define dpsi0(zFunc) (zFunc*zFunc * ( zFunc * (-30.0e0*zFunc + 60.0e0) - 30.0e0))
#define ddpsi0(zFunc) (zFunc* ( zFunc*( -120.0e0*zFunc + 180.0e0) -60.0e0))

// psi1 and its derivatives
#define psi1(zFunc) (zFunc*( zFunc*zFunc * ( zFunc * (-3.0e0*zFunc + 8.0e0) - 6.0e0) + 1.0e0))
#define dpsi1(zFunc) (zFunc*zFunc * ( zFunc * (-15.0e0*zFunc + 32.0e0) - 18.0e0) +1.0e0)
#define ddpsi1(zFunc) (zFunc * (zFunc * (-60.0e0*zFunc + 96.0e0) -36.0e0))

// psi2  and its derivatives
#define psi2(zFunc) (0.5e0*zFunc*zFunc*( zFunc* ( zFunc * (-zFunc + 3.0e0) - 3.0e0) + 1.0e0))
#define dpsi2(zFunc) (0.5e0*zFunc*( zFunc*(zFunc*(-5.0e0*zFunc + 12.0e0) - 9.0e0) + 2.0e0))
#define ddpsi2(zFunc) (0.5e0*(zFunc*( zFunc * (-20.0e0*zFunc + 36.0e0) - 18.0e0) + 2.0e0))

#define h5(w0t, w1t, w2t, w0mt, w1mt, w2mt, w0d, w1d, w2d, w0md, w1md, w2md, fi) (fi[0]  *w0d*w0t   + fi[1]  *w0md*w0t  + fi[2]  *w0d*w0mt  + fi[3]  *w0md*w0mt + fi[4]  *w0d*w1t   + fi[5]  *w0md*w1t + fi[6]  *w0d*w1mt  + fi[7]  *w0md*w1mt + fi[8]  *w0d*w2t   + fi[9] *w0md*w2t + fi[10] *w0d*w2mt  + fi[11] *w0md*w2mt + fi[12] *w1d*w0t   + fi[13] *w1md*w0t + fi[14] *w1d*w0mt  + fi[15] *w1md*w0mt  + fi[16] *w2d*w0t   + fi[17] *w2md*w0t + fi[18] *w2d*w0mt  + fi[19] *w2md*w0mt + fi[20] *w1d*w1t   + fi[21] *w1md*w1t + fi[22] *w1d*w1mt  + fi[23] *w1md*w1mt + fi[24] *w2d*w1t   + fi[25] *w2md*w1t + fi[26] *w2d*w1mt  + fi[27] *w2md*w1mt + fi[28] *w1d*w2t   + fi[29] *w1md*w2t + fi[30] *w1d*w2mt  + fi[31] *w1md*w2mt + fi[32] *w2d*w2t   + fi[33] *w2md*w2t + fi[34] *w2d*w2mt  + fi[35] *w2md*w2mt)

//  cubic hermite polynomial statement functions
//  psi0 & derivatives
#define xpsi0(zFunc) (zFunc * zFunc * (2.0e0*zFunc - 3.0e0) + 1.0)

//  psi1 & derivatives
#define xpsi1(zFunc) (zFunc * ( zFunc * (zFunc - 2.0e0) + 1.0e0))

/*************************************************************************
Section with derived quantities
*************************************************************************/
/** Grid size without AMR **/
#define N1  (NB_1*BS_1)
#define N2  (NB_2*BS_2)
#define N3  (NB_3*BS_3)

/*Set number of boundary cells in grid depending on order of spatial reconstruction*/
#define NG (2+PPM)
#define N1M ((N1>1)?(N1+2*NG):(1))
#define N2M ((N2>1)?(N2+2*NG):(1))
#define N3M ((N3>1)?(N3+2*NG):(1))

#define N1G ((N1>1)?(NG):(0))
#define N2G ((N2>1)?(NG):(0))
#define N3G ((N3>1)?(NG):(0))

#define D1 (N1>1)
#define D2 (N2>1)
#define D3 (N3>1)

/*Set variable numbers*/
#define NPR_U      (8+DOKTOT)        /* number of gas primitive variables */
#define NPR_R      (4)        /* number of radiation primitive variables */
#define NPR_2T     (2)        /* number of hydrodynamic primitive variables */
#define NPR_PH     (1)        /* Number density of photons*/
#define NPR_E      (3)        /* number of electric field primitive variables */
#define NPR_HD     (5)        /* number of hydrodynamic primitive variables */
#define NPR        (NPR_U+RAD_M1*NPR_R+RESISTIVE*NPR_E+TWO_T*NPR_2T+P_NUM*NPR_PH)        /* total number of primitive variables */
#define NDIM       (4)        /* number of total dimensions.  Never changes */
#define NPG        (5)        /* number of positions on grid for grid functions */
#define NSOLVER    (4)		/* number of positions on grid for HLLC and HLLD solver transformation matrix */
#define COMPDIM    (2)        /* number of non-trivial spatial dimensions used in computation */
#define NIMG       (4)        /* Number of types of images to make, kind of */
#define NFAIL	   (5)        /* Number of types of failure images to make*/

#define NPRDUMP (9+5*RAD_M1+2*TWO_T+3*RESISTIVE)

/*Based on derefinement level near pole set total number of AMR levels*/
#if(NB_2==6 && DEREFINE_POLE)
#define N_LEVELS_1D 1
#elif(NB_2 == 12 && DEREFINE_POLE)
#define N_LEVELS_1D 2
#elif(NB_2 == 24 && DEREFINE_POLE)
#define N_LEVELS_1D 3
#elif(NB_2 == 48 && DEREFINE_POLE)
#define N_LEVELS_1D 4
#elif(NB_2 == 96 && DEREFINE_POLE)
#define N_LEVELS_1D 5
#else
#define N_LEVELS_1D 0
#endif
#define N_LEVELS (N_LEVELS_1D+N_LEVELS_3D)

/*Calculate number of AMR blocks for different refinement levels and configurations*/
#if(REF_3+REF_2+REF_1==2)
#if (N_LEVELS==1)
#define NB (NB_1*NB_2*NB_3)
#elif(N_LEVELS==2)
#define NB (NB_1*NB_2*NB_3*(4+1))
#elif(N_LEVELS==3)
#define NB (NB_1*NB_2*NB_3*(4*(4+1)+1))
#elif(N_LEVELS==4)
#define NB (NB_1*NB_2*NB_3*(4*(4*(4+1)+1)+1))
#elif(N_LEVELS==5)
#define NB (NB_1*NB_2*NB_3*(4*(4*(4*(4+1)+1)+1)+1))
#endif
#elif(REF_3+REF_2+REF_1==3)
#if (N_LEVELS_3D==1)
#define FACTOR1 (1)
#define FACTOR2 (1)
#elif(N_LEVELS_3D==2)
#define FACTOR1 (8+1)
#define FACTOR2 ((6)+1)
#elif(N_LEVELS_3D==3)
#define FACTOR1 (8*8+8+1)
#define FACTOR2 ((4*8+2*6)+6+1)
#elif(N_LEVELS_3D==4)
#define FACTOR1 (8*8*8+8*8+8+1)
#define FACTOR2 ((4*8*8+2*(4*8+2*6))+4*8+2*6+6+1)
#elif(N_LEVELS_3D==5)
#define FACTOR1 (8*8*8*8+8*8*8+8*8+8+1)
#define FACTOR2 ((4*8*8*8+2*(4*8*8+2*(4*8+2*6)))+4*8*8+2*(4*8+2*6)+4*8+2*6+6+1)
#endif
#if (N_LEVELS_1D==0)
#define NB (NB_1*NB_2*NB_3*FACTOR1)
#elif (N_LEVELS_1D==1)
#define NB (NB_1*NB_3*(2*4*FACTOR1+2*(FACTOR2)+4))
#elif(N_LEVELS_1D==2)
#define NB (NB_1*NB_3*((4*8*FACTOR1)+(2*2*FACTOR1+2*8)+(2*(FACTOR2)+10)))
#elif(N_LEVELS_1D==3)
#define NB (NB_1*NB_3*((8*16*FACTOR1)+(4*4*FACTOR1+4*16)+(2*2*FACTOR1+2*20)+(2*(FACTOR2)+22)))
#elif(N_LEVELS_1D==4)
#define NB (NB_1*NB_3*((16*32*FACTOR1)+(8*8*FACTOR1+8*32)+(4*4*FACTOR1+4*40)+(2*2*FACTOR1+2*44)+(2*(FACTOR2)+46)))
#elif(N_LEVELS_1D==5)
#define NB (NB_1*NB_3*((32*64*FACTOR1)+(16*16*FACTOR1+16*64)+(8*8*FACTOR1+8*80)+(4*4*FACTOR1+4*88)+(2*2*FACTOR1+2*92)+(2*(FACTOR2)+94)))
#endif
#elif(REF_3+REF_2+REF_1==1)
#if (N_LEVELS==1)
#define NB (NB_1*NB_2*NB_3)
#elif(N_LEVELS==2)
#define NB (NB_1*NB_2*NB_3*(2+1))
#elif(N_LEVELS==3)
#define NB (NB_1*NB_2*NB_3*(2*(2+1)+1))
#elif(N_LEVELS==4)
#define NB (NB_1*NB_2*NB_3*(2*(2*(2+1)+1)+1))
#elif(N_LEVELS==5)
#define NB (NB_1*NB_2*NB_3*(2*(2*(2*(2+1)+1)+1)+1))
#endif
#endif

#if(HLLC==1 || HLLD==1)
#define FRAME_TRANSFORM (1)
#else
#define FRAME_TRANSFORM (0)
#endif

/*Define offset to make GPU memory access coalesced*/
#define FIX_MEM1 (LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE)
#define FIX_MEM2 (LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE)

/*Macro declerations*/
#define PLOOP  for(k=0;k<NPR;k++) //loop over all Dimensions; second rank loop */
#define DLOOP  for(j=0;j<NDIM;j++) for(k=0;k<NDIM;k++)//loop over all Dimensions; first rank loop */
#define DLOOPA for(j=0;j<NDIM;j++) //loop over all Space dimensions; second rank loop */
#define SLOOP  for(j=1;j<NDIM;j++) for(k=1;k<NDIM;k++) //loop over all Space dimensions; first rank loop */
#define SLOOPA for(j=1;j<NDIM;j++) // loop over Primitive variables 
#define MY_MIN(fval1,fval2) ( ((fval1) < (fval2)) ? (fval1) : (fval2))
#define MY_MAX(fval1,fval2) ( ((fval1) > (fval2)) ? (fval1) : (fval2))
#define delta(i,j) ( (i == j) ? 1. : 0.)
#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) 
#define ZLOOP for(i=0;i<N1;i++)for(j=0;j<N2;j++)
#define ZLOOP_MPI for(i=N1_GPU_offset[n_ord[n]];i<N1_GPU_offset[n_ord[n]] + BS_1;i++)for(j=N2_GPU_offset[n_ord[n]];j<N2_GPU_offset[n_ord[n]] + BS_2 ;j++)
#if (N3>1)
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_GPU_offset[n_ord[n]];i<N1_GPU_offset[n_ord[n]] + BS_1;i++)for(j=N2_GPU_offset[n_ord[n]];j<N2_GPU_offset[n_ord[n]] + BS_2 ;j++)for(z=N3_GPU_offset[n_ord[n]];z<N3_GPU_offset[n_ord[n]] + BS_3 ;z++)
#else
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_GPU_offset[n_ord[n]];i<N1_GPU_offset[n_ord[n]] + BS_1;i++)for(j=N2_GPU_offset[n_ord[n]];j<N2_GPU_offset[n_ord[n]] + BS_2 ;j++)for(z=N3_GPU_offset[n_ord[n]];z<N3_GPU_offset[n_ord[n]] + BS_3 ;z++)
#endif
#define ZSLOOP(istart,istop,jstart,jstop) for(i=istart;i<=istop;i++) for(j=jstart;j<=jstop;j++)
#if (N3>1)
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#define ZSLOOPZIRI(istart, istop, jstart, jstop, zstart, zstop) for(z=zstart;z<=zstop;z++) for (j = jstart; j <= jstop; j++) for (i = istart; i <= istop; i++)
#else
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#define ZSLOOPZIRI(istart, istop, jstart, jstop, zstart, zstop) for(z=zstart;z<=zstop;z++) for (j = jstart; j <= jstop; j++) for (i = istart; i <= istop; i++)
#endif