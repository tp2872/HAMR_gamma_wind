#include "u2p_defs.h"

extern void primtoU_g( FTYPE prim[], FTYPE gcov[][NDIM], FTYPE gcon[][NDIM], FTYPE gdet,  FTYPE U[] );
extern void ucon_calc_g(FTYPE prim[],FTYPE gcov[][NDIM],FTYPE gcon[][NDIM],FTYPE ucon[]);
extern void raise_g(FTYPE vcov[], FTYPE gcon[][NDIM], FTYPE vcon[]);
extern void lower_g(FTYPE vcon[], FTYPE gcov[][NDIM], FTYPE vcov[]);
extern void ncov_calc(FTYPE gcon[][NDIM],FTYPE ncov[]) ;
extern void bcon_calc_g(FTYPE prim[],FTYPE ucon[],FTYPE ucov[],FTYPE ncov[],FTYPE bcon[]); 
extern FTYPE pressure_rho0_u(FTYPE rho0, FTYPE u);
extern FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w);
extern FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w);
extern int gamma_calc_g(FTYPE *pr, FTYPE gcov[NDIM][NDIM], FTYPE *gamma);

