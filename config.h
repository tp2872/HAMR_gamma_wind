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
#define WHICHPROBLEM TORUS_PROBLEM

/*Enable special refinement criterion for large scale jet simulations*/
#define REFINE_JET (0)

/*Select adiabatic index and BH spin*/
#define GAMMA	(5./3.)
#define BH_SPIN (0.9375)

/*Wheter or not to tilt the disk*/
#define TILTED (0)
#define TILT_ANGLE (0.)

/*Wheter to activate an untilted elliptical disk*/
#define ELLIPTICAL (0)
#define ELLIPTICAL2 (0)

/*Wheter to cool the disk to predifined thickness H_OVER_R. Not implemented in CPU version*/
#define COOL_DISK (0)
#define H_OVER_R (0.03)

/*Wheter or not to use the full dispersion relation. Only slows down simulation and does not really increase accuracy*/
#define FULL_DISP (0)

/** FIXUP PARAMETERS, magnitudes of rho and u, respectively, in the floor : **/
#define RHOMIN	(1.e-6)
#define UUMIN	(1.e-7)
#define RHOMINLIMIT (1.e-20)
#define UUMINLIMIT  (1.e-20)
#define POWRHO (1.5)
#define FLOORFACTOR (1.0)
#define BSQORHOMAX (20.*FLOORFACTOR)
#define BSQOUMAX (750.*FLOORFACTOR)
#define UORHOMAX (150.*FLOORFACTOR)

/* Max. value of gamma, the lorentz factor */
#define GAMMAMAX (80.)

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
#define AMD (1)

/*Enable if running on the new VOLTA GPUs*/
#define V100 (1)

/*Use NVIDIA GPU_DIRECT. Check availability on cluster and enable it in slurm job script, for mpich set MPICH_RDMA_ENABLED_CUDA=1*/
#define GPU_DIRECT 1

/*Maximum tag number for MPI messages so not to overflow*/
#define MPI_TAG_MAX 1264576

/*Determine if you want to explicitely copy the B fields from block to block. Good to use when working on AMR, since a good implementation gives divB=0*/
#define COPY_BFIELD 1

/*Maximum number of blocks per node and hten umber of memory places(should be equal)*/
#define MAX_BLOCKS (1600)
#define NB_LOCAL (1600)

/*Define number of blocks for the first AMR level in all three dimensions*/
#define NB_1 1
#define NB_2 4
#define NB_3 1

/*Set block size in each dimension*/
#define BS_1 128
#define BS_2 32
#define BS_3 128

/*Set the maximum number of refinement levels*/
#define N_LEVELS_3D 1

/*Set in which dimensions to refine for AMR. You must set at least one value to 1 for the correct functioning of the code*/
#define REF_1 1
#define REF_2 1
#define REF_3 1

/*Number of GPUs per MPI rank*/
#define N_GPU 1

/*Derefines the pole in the third dimension. Make sure REF_3==1 and NB_2=6,12,24,48 and NB_1=4 and NB_3>=2*/
#define DEREFINE_POLE (0)

/*Number of internal derefinement levels*/
#define N_LEVELS_1D_INT (0)

/*Enable very fast hierarchical timestepping routine in combination with DEREFINE_POLE and REF_1=0, REF_2=0, REF_3=1. Requires that number NB2*NB3/NUM_OF_MPI_PROCESSES is an integer number*/
#define TIMESTEP_JET 0

//Use Z-order at 0-level for load balancing
#define Z_ORDER 0

/*Set maximum timelevel for AMR (ie 1,2,4,8 etc). This determines how often the timestep is changed so setting it to an absurd high value may cause code crashes
If a very high value is needed, lowerin Courant factor may increase stability*/
#define AMR_MAXTIMELEVEL 16

/*The minimum timeinterval at which refinement takes place, TREF can't go below it*/
#define AMR_SWITCHTIMELEVEL 32

/*Use prestepping for load balancing with HTS*/
#define PRESTEP 0

/*Use second order timestepping at LAS boundaries, not possible in combination with PRESTEP*/
#define PRESTEP2 0

/*Preevolve primitive variables for Summit*/
#define PRESTEP_P 0

/*Used for loading in old data files. Do not touch!*/
#define REVERSE_ORDERING 0

//The time between refinement (AMR) steps
#define TREF 10.

/*Select the courant factor for the timestep*/
#define COUR (0.9)

/*Evolve entropy for more stability*/
#define DO_FONT_FIX (1) //Use redundant inversion scheme for more stability
#define DOKTOT 1  //Evolve entropy to do the above even more accurately

/*Enable/disable PPM/van Leer spatial reconstruction. Never enable both*/
#define PPM (1)
#define LEER (0)

/*Wheter to set floors in ZAMO frame*/
#define ZAMO_FLOOR (0)

/*Wheter to set floors in drift frame*/
#define DRIFT_FLOOR (1)

/*Whether or not to allow inflow for fluxes (see fix_flux())*/
#define INFLOW 0

/*Enable or disable the HLLC solver. Does not work yet!*/
#define HLLC (0)

/*Whether or not to use a staggered grid*/
#define STAGGERED (1)

/* use local lax-friedrichs or HLL flux:  these are relative weights on each numerical flux */
#define HLLF  (1)
#define LAXF  (0)

/*Wheter or not to use a non symmetric metric for tilted disk. Not fully implemented in this version!*/
#define NSY (0)

/*Use transmissive boundary condition at pole*/
#define TRANS_BOUND (1)

/* how many cells near the poles to stabilize, choose 0 for no stabilization */
#define POLEFIX 2

/*Set periodic boundary conditions only in the third dimension is supported*/
#define PERIODIC1 0
#define PERIODIC2 0
#if (BS_3*NB3==1)
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

/*Wheter to cylindrify coordinates to increase GLOBAL timestep*/
#define DOCYLINDRIFYCOORDS (0)

/*Put out files which Ziri can Ray-Trace. Not fully implemented yet*/
#define ZIRI_DUMP 0

/* whether or not to rescale primitive variables before interpolating them for flux/BC's. Is not implemented on GPU */
#define RESCALE     (0)

/*Enable MPI; Old remnant do not touch!*/
#define MPI_enable 1

/*Do not use; Used to load in 3 level AMR data into 4-level AMR grid*/
#define READ_OLD (0)

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
#define INIT_OUT	(0)
#define DUMP_OUT	(1)
#define IMAGE_OUT	(2)
#define LOG_OUT		(3)
#define FINAL_OUT	(4)

/* failure modes */
#define FAIL_UTOPRIM        (1)
#define FAIL_VCHAR_DISCR    (2)
#define FAIL_COEFF_NEG	    (3)
#define FAIL_COEFF_SUP	    (4)
#define FAIL_GAMMA          (5)
#define FAIL_METRIC         (6)

/* For rescale() operations: */
#define FORWARD 1
#define REVERSE 2

/*For Windows users*/
#ifndef M_PI 
#define M_PI 3.14159265358979323846264338327950288 
#endif 

/*Mnemonics for AMR parameters*/
#define NV 181
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
#define NPR        (8+DOKTOT)        /* number of primitive variables */
#define NDIM       (4)        /* number of total dimensions.  Never changes */
#define NPG        (5)        /* number of positions on grid for grid functions */
#define COMPDIM    (2)        /* number of non-trivial spatial dimensions used in computation */
#define NIMG       (4)        /* Number of types of images to make, kind of */
#define NFAIL	   (5)        /* Number of types of failure images to make*/

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