#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* numerical parameters */
extern double Rin,Rout,hslope,R0,fractheta ;
extern double cour;
extern int lim;

/* physics parameters */
extern double a ;
extern double gam ;
#define NPR        (8)        /* number of primitive variables */
#define NDIM       (4)        /* number of total dimensions.  Never changes */
#define NPG        (5)        /* number of positions on grid for grid functions */
#define COMPDIM    (2)        /* number of non-trivial spatial dimensions used in computation */
#define NIMG       (4)        /* Number of types of images to make, kind of */
#define NFAIL	   (5)        /* Number of types of failure images to make*/

/*Set freezing point*/
#define ibound 0
#define jbound 0
#define zbound 0

#define boundfreeze1 0 //Movie playback(not yet fully implemented)
#define boundfreeze2 0 //Frozen boundary at i=ibound-1
#if boundfreeze2==1
#define N_POINTS 2
#elif boundfreeze1==1
#define N_POINTS 10
#else
#define N_POINTS 1
#endif
/*************************************************************************
    MACROS
*************************************************************************/
/* loop over all active zones */
#define ZLOOP for(i=0;i<N1;i++)for(j=0;j<N2;j++)
#define ZLOOP_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)
#if (N3>1)
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)for(z=N3_MPI_offset;z<N3_MPI_offset + N3_MPI ;z++)
#else
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)for(z=N3_MPI_offset;z<N3_MPI_offset + N3_MPI ;z++)
#endif

/* loop over all active zones */
#define IMAGELOOP for(j=0;j<N2;j++)for(i=0;i<N1;i++)

/* specialty loop */
#define ZSLOOP(istart,istop,jstart,jstop) for(i=istart;i<=istop;i++) for(j=jstart;j<=jstop;j++)
#if (N3>1)
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#else
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#endif


/* grid functions */
extern double(*   p)[NPR];
extern double(*   ph)[NPR];
extern double(*  dq)[NPR];
extern double(*  F1)[NPR];
extern double(*  F2)[NPR];
extern double(*  F3)[NPR];
extern int(*  pflag);
extern double (* conn)[NDIM][NDIM][NDIM] ;
extern double (* gcon)[NPG][NDIM][NDIM] ;
extern double (* gcov)[NPG][NDIM][NDIM] ;
extern double (* gdet)[NPG];
extern int(* failimage)[NFAIL];
extern double Katm[2000];

/*MPI variables*/
extern int n_rows;
extern int n_columns;
extern int n_stacks;
extern int n1_MPI;
extern int n2_MPI;
extern int n3_MPI;
extern int N1_MPI;
extern int N2_MPI;
extern int N3_MPI;
extern int N1_MPI_offset;
extern int N2_MPI_offset;
extern int N3_MPI_offset;
extern int numtasks, rank;
extern int N1_tot;
extern int N2_tot;
extern int N3_tot;

/*CUDA variables*/
extern int fix_mem;
extern int fix_mem2;
extern int nr_workgroups;
extern size_t global_work_size[1];
extern size_t global_work_size_bound[1];
extern size_t global_work_offset[1];
extern size_t global_work_size_special[1];
extern size_t global_work_size_special1[1];
extern size_t global_work_size_special2[1];
extern size_t global_work_size_special3[1];
extern size_t local_work_size[1];
extern double* BufferF1_1;
extern double* BufferF2_1;
extern double* BufferF3_1;
extern double* Bufferdq_1;
extern double* Bufferstorage1;
extern double* Bufferstorage2;
extern double* Bufferstorage3;
extern double* Bufferstorage4;
extern double* Bufferp_1;
extern double* Bufferph_1;
extern double* Bufferpbound_1;
extern double* Buffergcov;
extern double* Buffergcon;
extern double* Bufferconn;
extern double* Buffergdet;
extern double* Bufferdtij;
extern double* Bufferdiagflux;
extern int* Bufferpflag;
extern int* Bufferfailimage;
extern double* BufferKatm;
extern int* BufferaN1_MPI;
extern int* BufferaN1_MPI_offset;
extern int* BufferaN2_MPI;
extern int* BufferaN2_MPI_offset;
extern int* BufferaN3_MPI;
extern int* BufferaN3_MPI_offset;
extern int* Buffern1_MPI;
extern int* Buffern2_MPI;
extern int* Buffern3_MPI;
extern int* Bufferrank;
extern double* Bufferboundrec1_MPI;
extern double* Bufferboundrec2_MPI;
extern double* Bufferboundrec3_MPI;
extern double* Bufferboundrec4_MPI;
extern double* Bufferboundrec5_MPI;
extern double* Bufferboundrec6_MPI;
extern double* Bufferboundrec7_MPI;
extern double* Bufferboundrec8_MPI;
extern double* Bufferboundsend1_MPI;
extern double* Bufferboundsend2_MPI;
extern double* Bufferboundsend3_MPI;
extern double* Bufferboundsend4_MPI;
extern double* Bufferboundsend5_MPI;
extern double* Bufferboundsend6_MPI;
extern double* Bufferboundsend7_MPI;
extern double* Bufferboundsend8_MPI;
extern double* Buffercornrec1_MPI;
extern double* Buffercornrec2_MPI;
extern double* Buffercornrec3_MPI;
extern double* Buffercornrec4_MPI;
extern double* Buffercornrec5_MPI;
extern double* Buffercornrec6_MPI;
extern double* Buffercornrec7_MPI;
extern double* Buffercornrec8_MPI;
extern double* Buffercornsend1_MPI;
extern double* Buffercornsend2_MPI;
extern double* Buffercornsend3_MPI;
extern double* Buffercornsend4_MPI;
extern double* Buffercornsend5_MPI;
extern double* Buffercornsend6_MPI;
extern double* Buffercornsend7_MPI;
extern double* Buffercornsend8_MPI;
extern int *aN1_MPI_offset;
extern int *aN2_MPI_offset;
extern int *aN3_MPI_offset;
extern int *aN1_MPI;
extern int *aN2_MPI;
extern int *aN3_MPI;
 
/*GPU variables*/
#define FTYPE2 double
extern FTYPE2 *F1_1;
extern FTYPE2 *F2_1;
extern FTYPE2 *F3_1;
extern FTYPE2 *dq_1;
extern FTYPE2 *p_1;
extern FTYPE2 *ph_1;
extern FTYPE2 *pbound_1;
extern FTYPE2 *gcov_GPU;
extern FTYPE2 *gcon_GPU;
extern FTYPE2 *conn_GPU;
extern FTYPE2 *gdet_GPU;
extern FTYPE2 *dtij_GPU;
extern FTYPE2 *Katm_GPU;
extern int *pflag_GPU;
extern int *failimage_GPU;

/*Function definitions*/
double fluxcalc_GPU(int dir, int flag);
void GPU_init(void);
void GPU_write(void);
void GPU_fluxcalcprep(int dir,int flag);
void GPU_fluxcalc2D(int dir, int flag);
void GPU_fix_flux(void);
void GPU_flux_ct1(void);
void GPU_flux_ct2(void);
void GPU_diag_flux(void);
void GPU_Utoprim(int flag, double Dt);
void GPU_fixuputoprim(int flag);
void GPU_fixup(int flag);
void GPU_boundprim(int flag,int MPI);
void GPU_boundprim1(int flag);
void GPU_boundprim2(int flag);
void GPU_boundprim3(int flag);
void GPU_boundsend(int flag);
void GPU_boundrec(int flag);
void GPU_read(void);
int index(int i, int j, int z);
int index2(int i, int j);
int index3(int i, int j);