#include <mpi.h>
#include "decs.h"
extern MPI_Request req[100], boundreqs[NB][600], cornreqs[NB][16];
extern MPI_Status Statbound[NB][600], Statcorn[NB][16], Statrec[2];
extern MPI_Comm  mpi_cartcomm, mpi_self;
extern MPI_Comm row_comm[8];
extern MPI_File fdump[100], fdumpdiag[100], rdump[NB];
extern MPI_Request req_block[NB][1];
extern MPI_Request req_block_rdump[NB][1];
extern MPI_Request req_blockdiag[NB][1];
extern MPI_Request request_timelevel[NB];
void dump_block(MPI_File *fp, int n);
void dump_blockdiag(MPI_File *fp, int n);
void rdump_block_write(MPI_File *fp, int n);

