#include "decs.h" 
#include <mpi.h>

//MPI Variables
extern MPI_Request req[NB_LOCAL], boundreqs[NB_LOCAL][600];
extern MPI_Status Statbound[NB_LOCAL][600];
extern MPI_Comm  mpi_cartcomm, mpi_self;
extern MPI_Comm row_comm[8];
extern MPI_File fdump[100], fdumpdiag[100], rdump[NB_LOCAL],gdump[NB_LOCAL];
extern MPI_Request req_block[NB_LOCAL][1], req_block_rdump[NB_LOCAL][1], req_blockdiag[NB_LOCAL][1], req_gdump1[NB_LOCAL][1], req_gdump2[NB_LOCAL][1];
extern MPI_Request request_timelevel[NB];

//MPI functions
void dump_block(MPI_File *fp, int n);
void dump_blockdiag(MPI_File *fp, int n);
void rdump_block_write(MPI_File *fp, int n);
void gdump_block(MPI_File *fp, int n);

