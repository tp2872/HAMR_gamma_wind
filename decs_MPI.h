#include "decs.h" 
#include <mpi.h>

//MPI Variables
extern MPI_Request req[NB], boundreqs[NB_LOCAL][600];
extern MPI_Status Statbound[NB_LOCAL][600];
extern MPI_Comm  mpi_cartcomm, mpi_self;
extern MPI_Comm row_comm[8];
extern MPI_File fdump[2000], fdump_reduced[2000], fdumpdiag[2000], rdump[NB_LOCAL], gdump[NB_LOCAL], gdump_reduced[NB_LOCAL], grid_dump[1], grid_restart[1];
extern MPI_Request req_block[NB_LOCAL][1], req_block_reduced[NB_LOCAL][1], req_block_rdump[NB_LOCAL][1], req_blockdiag[NB_LOCAL][1], req_gdump1[NB_LOCAL][1], req_gdump2[NB_LOCAL][1], req_gdump1_reduced[NB_LOCAL][1], req_gdump2_reduced[NB_LOCAL][1], req_gdumpgrid[1], req_rdumpgrid[1];
extern MPI_Request request_timelevel[NB];
extern MPI_Request req_local1[N_LEVELS_3D][NB_1*NB_3 * 64], req_local2[N_LEVELS_3D][NB_1*NB_3 * 64];
extern int send_tag1[N_LEVELS_3D][MY_MAX(NB, 60000)], send_tag2[N_LEVELS_3D][MY_MAX(NB, 60000)];

//MPI functions
void dump_block(MPI_File *fp, int n);
void dump_block_reduced(MPI_File *fp, int n);
void dump_blockdiag(MPI_File *fp, int n);
void rdump_block_write(MPI_File *fp, int n);
void gdump_block(MPI_File *fp, int n);
void gdump_block_reduced(MPI_File  *fp, int n);
void gdump_grid(FILE *fp);
void rdump_grid(MPI_File *fp);

