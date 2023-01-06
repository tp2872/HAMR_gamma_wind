#ifndef __APPLE__
#include <malloc.h>
#endif

/*************************************************************************
GLOBAL ARRAY SECTION
*************************************************************************/
/* for debug */
int(*restrict failimage[NB_LOCAL])[NFAIL];
#if(DO_FONT_FIX)
double *Katm[NB_LOCAL];
#endif

/*CPU arrays*/
double(*restrict V[NB_LOCAL])[6];
double(*restrict p[NB_LOCAL])[NPR];
double(*E_avg1[N_LEVELS_3D])[BS_1 + 2 * N1G];
double(*E_avg2[N_LEVELS_3D])[BS_1 + 2 * N1G];
double(*E_avg1_new[N_LEVELS_3D])[BS_1 + 2 * N1G];
double(*E_avg2_new[N_LEVELS_3D])[BS_1 + 2 * N1G];
double(*restrict ph[NB_LOCAL])[NPR];
double(*restrict E_corn[NB_LOCAL])[NDIM];
double(*restrict dE[NB_LOCAL])[2][NDIM][NDIM];
double(*restrict ps[NB_LOCAL])[NDIM];
double(*restrict psh[NB_LOCAL])[NDIM];
double(*restrict U_n[NB_LOCAL])[NPR];
double(*restrict U_0[NB_LOCAL])[NPR];
double(*restrict U_1[NB_LOCAL])[NPR];
double(*restrict U[NB_LOCAL])[NPR];
double(*restrict dU_MHD1[NB_LOCAL])[NPR];
double(*restrict dU_RAD0[NB_LOCAL])[NPR];
double(*restrict dU_RAD1[NB_LOCAL])[NPR];
double(*restrict dq[NB_LOCAL])[NPR];
double(*restrict F1[NB_LOCAL])[NPR];
double(*restrict F2[NB_LOCAL])[NPR];
double(*restrict F3[NB_LOCAL])[NPR];
int(*restrict pflag[NB_LOCAL]);
int(*restrict pflag_cart[NB_LOCAL]);
int(*restrict pflag_rad[NB_LOCAL]);
double(*restrict conn[NB_LOCAL])[NDIM][NDIM][NDIM];
double(*restrict gcon[NB_LOCAL])[NPG][NDIM][NDIM];
double(*restrict gcov[NB_LOCAL])[NPG][NDIM][NDIM];
double(*restrict gdet[NB_LOCAL])[NPG];
double(*restrict Mud[NB_LOCAL])[NDIM][NDIM][NDIM];
double(*restrict Mud_inv[NB_LOCAL])[NDIM][NDIM][NDIM];
double(*restrict dU_s[NB_LOCAL])[NPR];

/*EOS CPU arrays*/
double eos_f[EOSIMAX*EOSJMAX];
double eos_fd[EOSIMAX*EOSJMAX];
double eos_ft[EOSIMAX*EOSJMAX];
double eos_fdd[EOSIMAX*EOSJMAX];
double eos_ftt[EOSIMAX*EOSJMAX];
double eos_fdt[EOSIMAX*EOSJMAX];
double eos_fddt[EOSIMAX*EOSJMAX];
double eos_fdtt[EOSIMAX*EOSJMAX];
double eos_fddtt[EOSIMAX*EOSJMAX];
double eos_dpdf[EOSIMAX*EOSJMAX];
double eos_dpdfd[EOSIMAX*EOSJMAX];
double eos_dpdft[EOSIMAX*EOSJMAX];
double eos_dpdfdt[EOSIMAX*EOSJMAX];
double eos_ef[EOSIMAX*EOSJMAX];
double eos_efd[EOSIMAX*EOSJMAX];
double eos_eft[EOSIMAX*EOSJMAX];
double eos_efdt[EOSIMAX*EOSJMAX];
double eos_xf[EOSIMAX*EOSJMAX];
double eos_xfd[EOSIMAX*EOSJMAX];
double eos_xft[EOSIMAX*EOSJMAX];
double eos_xfdt[EOSIMAX*EOSJMAX];
double eos_t[EOSJMAX];
double eos_d[EOSIMAX];
double eos_dd[EOSIMAX];
double eos_dt[EOSJMAX];

/*GPU arrays*/
double *F1_1[NB_LOCAL];
double *F2_1[NB_LOCAL];
double *F3_1[NB_LOCAL];
double *dq_1[NB_LOCAL];
double *dU_GPU[NB_LOCAL];
double *p_1[NB_LOCAL];
double *ph_1[NB_LOCAL];
double *ps_1[NB_LOCAL];
double *psh_1[NB_LOCAL];
double *gcov_GPU[NB_LOCAL];
double *gcon_GPU[NB_LOCAL];
double *Mud_GPU[NB_LOCAL];
double *Mud_inv_GPU[NB_LOCAL];
double *conn_GPU[NB_LOCAL];
double *gdet_GPU[NB_LOCAL];
double *dtij1_GPU[NB_LOCAL];
double *dtij2_GPU[NB_LOCAL];
double *dtij3_GPU[NB_LOCAL];
double *Katm_GPU[NB_LOCAL];
int *pflag_GPU[NB_LOCAL];
int* pflag_CART_GPU[NB_LOCAL];
int *failimage_GPU[NB_LOCAL];
double* radius_GPU[NB_LOCAL];
int failimage_counter[NFAIL];

/*MPI arrays*/
double  *send1[NB_LOCAL], *send2[NB_LOCAL], *send3[NB_LOCAL], *send4[NB_LOCAL], *send5[NB_LOCAL], *send6[NB_LOCAL];
double  *send1_fine[NB_LOCAL], *send2_fine[NB_LOCAL], *send3_fine[NB_LOCAL], *send4_fine[NB_LOCAL], *send5_fine[NB_LOCAL], *send6_fine[NB_LOCAL];

double  *send1_flux[NB_LOCAL], *send2_flux[NB_LOCAL], *send3_flux[NB_LOCAL], *send4_flux[NB_LOCAL], *send5_flux[NB_LOCAL], *send6_flux[NB_LOCAL], *send7_flux[NB_LOCAL], *send8_flux[NB_LOCAL];
double  *send1_E[NB_LOCAL], *send2_E[NB_LOCAL], *send3_E[NB_LOCAL], *send4_E[NB_LOCAL], *send5_E[NB_LOCAL], *send6_E[NB_LOCAL], *send7_E[NB_LOCAL], *send8_E[NB_LOCAL];

double  *send1_3[NB_LOCAL], *send1_4[NB_LOCAL], *send1_7[NB_LOCAL], *send1_8[NB_LOCAL];
double  *receive1_3[NB_LOCAL], *receive1_4[NB_LOCAL], *receive1_7[NB_LOCAL], *receive1_8[NB_LOCAL];
double  *tempreceive1_3[NB_LOCAL], *tempreceive1_4[NB_LOCAL], *tempreceive1_7[NB_LOCAL], *tempreceive1_8[NB_LOCAL];

double  *send2_1[NB_LOCAL], *send2_2[NB_LOCAL], *send2_3[NB_LOCAL], *send2_4[NB_LOCAL];
double  *receive2_1[NB_LOCAL], *receive2_2[NB_LOCAL], *receive2_3[NB_LOCAL], *receive2_4[NB_LOCAL];
double  *tempreceive2_1[NB_LOCAL], *tempreceive2_2[NB_LOCAL], *tempreceive2_3[NB_LOCAL], *tempreceive2_4[NB_LOCAL];

double  *send3_1[NB_LOCAL], *send3_2[NB_LOCAL], *send3_5[NB_LOCAL], *send3_6[NB_LOCAL];
double  *receive3_1[NB_LOCAL], *receive3_2[NB_LOCAL], *receive3_5[NB_LOCAL], *receive3_6[NB_LOCAL];
double  *tempreceive3_1[NB_LOCAL], *tempreceive3_2[NB_LOCAL], *tempreceive3_5[NB_LOCAL], *tempreceive3_6[NB_LOCAL];

double  *send4_5[NB_LOCAL], *send4_6[NB_LOCAL], *send4_7[NB_LOCAL], *send4_8[NB_LOCAL];
double  *receive4_5[NB_LOCAL], *receive4_6[NB_LOCAL], *receive4_7[NB_LOCAL], *receive4_8[NB_LOCAL];
double  *tempreceive4_5[NB_LOCAL], *tempreceive4_6[NB_LOCAL], *tempreceive4_7[NB_LOCAL], *tempreceive4_8[NB_LOCAL];

double  *send5_1[NB_LOCAL], *send5_3[NB_LOCAL], *send5_5[NB_LOCAL], *send5_7[NB_LOCAL];
double  *receive5_1[NB_LOCAL], *receive5_3[NB_LOCAL], *receive5_5[NB_LOCAL], *receive5_7[NB_LOCAL];
double  *tempreceive5_1[NB_LOCAL], *tempreceive5_3[NB_LOCAL], *tempreceive5_5[NB_LOCAL], *tempreceive5_7[NB_LOCAL];

double  *send6_2[NB_LOCAL], *send6_4[NB_LOCAL], *send6_6[NB_LOCAL], *send6_8[NB_LOCAL];
double  *receive6_2[NB_LOCAL], *receive6_4[NB_LOCAL], *receive6_6[NB_LOCAL], *receive6_8[NB_LOCAL];
double  *tempreceive6_2[NB_LOCAL], *tempreceive6_4[NB_LOCAL], *tempreceive6_6[NB_LOCAL], *tempreceive6_8[NB_LOCAL];

double *receive1[NB_LOCAL], *receive2[NB_LOCAL], *receive3[NB_LOCAL], *receive4[NB_LOCAL], *receive5[NB_LOCAL], *receive6[NB_LOCAL];
double *tempreceive1[NB_LOCAL], *tempreceive2[NB_LOCAL], *tempreceive3[NB_LOCAL], *tempreceive4[NB_LOCAL], *tempreceive5[NB_LOCAL], *tempreceive6[NB_LOCAL];

double *receive1_fine[NB_LOCAL], *receive2_fine[NB_LOCAL], *receive3_fine[NB_LOCAL], *receive4_fine[NB_LOCAL], *receive5_fine[NB_LOCAL], *receive6_fine[NB_LOCAL];
double  *receive1_3fine[NB_LOCAL], *receive1_4fine[NB_LOCAL], *receive1_7fine[NB_LOCAL], *receive1_8fine[NB_LOCAL];
double  *receive2_1fine[NB_LOCAL], *receive2_2fine[NB_LOCAL], *receive2_3fine[NB_LOCAL], *receive2_4fine[NB_LOCAL];
double  *receive3_1fine[NB_LOCAL], *receive3_2fine[NB_LOCAL], *receive3_5fine[NB_LOCAL], *receive3_6fine[NB_LOCAL];
double  *receive4_5fine[NB_LOCAL], *receive4_6fine[NB_LOCAL], *receive4_7fine[NB_LOCAL], *receive4_8fine[NB_LOCAL];
double  *receive5_1fine[NB_LOCAL], *receive5_3fine[NB_LOCAL], *receive5_5fine[NB_LOCAL], *receive5_7fine[NB_LOCAL];
double  *receive6_2fine[NB_LOCAL], *receive6_4fine[NB_LOCAL], *receive6_6fine[NB_LOCAL], *receive6_8fine[NB_LOCAL];
double *receive1_flux[NB_LOCAL], *receive2_flux[NB_LOCAL], *receive3_flux[NB_LOCAL], *receive4_flux[NB_LOCAL], *receive5_flux[NB_LOCAL], *receive6_flux[NB_LOCAL], *receive7_flux[NB_LOCAL], *receive8_flux[NB_LOCAL];
double *receive1_flux1[NB_LOCAL], *receive2_flux1[NB_LOCAL], *receive3_flux1[NB_LOCAL], *receive4_flux1[NB_LOCAL], *receive5_flux1[NB_LOCAL], *receive6_flux1[NB_LOCAL], *receive7_flux1[NB_LOCAL], *receive8_flux1[NB_LOCAL];

double  *receive1_3flux[NB_LOCAL], *receive1_4flux[NB_LOCAL], *receive1_7flux[NB_LOCAL], *receive1_8flux[NB_LOCAL];
double  *receive2_1flux[NB_LOCAL], *receive2_2flux[NB_LOCAL], *receive2_3flux[NB_LOCAL], *receive2_4flux[NB_LOCAL];
double  *receive3_1flux[NB_LOCAL], *receive3_2flux[NB_LOCAL], *receive3_5flux[NB_LOCAL], *receive3_6flux[NB_LOCAL];
double  *receive4_5flux[NB_LOCAL], *receive4_6flux[NB_LOCAL], *receive4_7flux[NB_LOCAL], *receive4_8flux[NB_LOCAL];
double  *receive5_1flux[NB_LOCAL], *receive5_3flux[NB_LOCAL], *receive5_5flux[NB_LOCAL], *receive5_7flux[NB_LOCAL];
double  *receive6_2flux[NB_LOCAL], *receive6_4flux[NB_LOCAL], *receive6_6flux[NB_LOCAL], *receive6_8flux[NB_LOCAL];

double  *receive1_3flux1[NB_LOCAL], *receive1_4flux1[NB_LOCAL], *receive1_7flux1[NB_LOCAL], *receive1_8flux1[NB_LOCAL];
double  *receive2_1flux1[NB_LOCAL], *receive2_2flux1[NB_LOCAL], *receive2_3flux1[NB_LOCAL], *receive2_4flux1[NB_LOCAL];
double  *receive3_1flux1[NB_LOCAL], *receive3_2flux1[NB_LOCAL], *receive3_5flux1[NB_LOCAL], *receive3_6flux1[NB_LOCAL];
double  *receive4_5flux1[NB_LOCAL], *receive4_6flux1[NB_LOCAL], *receive4_7flux1[NB_LOCAL], *receive4_8flux1[NB_LOCAL];
double  *receive5_1flux1[NB_LOCAL], *receive5_3flux1[NB_LOCAL], *receive5_5flux1[NB_LOCAL], *receive5_7flux1[NB_LOCAL];
double  *receive6_2flux1[NB_LOCAL], *receive6_4flux1[NB_LOCAL], *receive6_6flux1[NB_LOCAL], *receive6_8flux1[NB_LOCAL];

double  *receive1_3flux2[NB_LOCAL], *receive1_4flux2[NB_LOCAL], *receive1_7flux2[NB_LOCAL], *receive1_8flux2[NB_LOCAL];
double  *receive2_1flux2[NB_LOCAL], *receive2_2flux2[NB_LOCAL], *receive2_3flux2[NB_LOCAL], *receive2_4flux2[NB_LOCAL];
double  *receive3_1flux2[NB_LOCAL], *receive3_2flux2[NB_LOCAL], *receive3_5flux2[NB_LOCAL], *receive3_6flux2[NB_LOCAL];
double  *receive4_5flux2[NB_LOCAL], *receive4_6flux2[NB_LOCAL], *receive4_7flux2[NB_LOCAL], *receive4_8flux2[NB_LOCAL];
double  *receive5_1flux2[NB_LOCAL], *receive5_3flux2[NB_LOCAL], *receive5_5flux2[NB_LOCAL], *receive5_7flux2[NB_LOCAL];
double  *receive6_2flux2[NB_LOCAL], *receive6_4flux2[NB_LOCAL], *receive6_6flux2[NB_LOCAL], *receive6_8flux2[NB_LOCAL];

double *receive1_E[NB_LOCAL], *receive2_E[NB_LOCAL], *receive3_E[NB_LOCAL], *receive4_E[NB_LOCAL], *receive5_E[NB_LOCAL], *receive6_E[NB_LOCAL], *receive7_E[NB_LOCAL], *receive8_E[NB_LOCAL];
double *receive1_E1[NB_LOCAL], *receive2_E1[NB_LOCAL], *receive3_E1[NB_LOCAL], *receive4_E1[NB_LOCAL], *receive5_E1[NB_LOCAL], *receive6_E1[NB_LOCAL], *receive7_E1[NB_LOCAL], *receive8_E1[NB_LOCAL];

double  *receive1_3E[NB_LOCAL], *receive1_4E[NB_LOCAL], *receive1_7E[NB_LOCAL], *receive1_8E[NB_LOCAL];
double  *receive2_1E[NB_LOCAL], *receive2_2E[NB_LOCAL], *receive2_3E[NB_LOCAL], *receive2_4E[NB_LOCAL];
double  *receive3_1E[NB_LOCAL], *receive3_2E[NB_LOCAL], *receive3_5E[NB_LOCAL], *receive3_6E[NB_LOCAL];
double  *receive4_5E[NB_LOCAL], *receive4_6E[NB_LOCAL], *receive4_7E[NB_LOCAL], *receive4_8E[NB_LOCAL];
double  *receive5_1E[NB_LOCAL], *receive5_3E[NB_LOCAL], *receive5_5E[NB_LOCAL], *receive5_7E[NB_LOCAL];
double  *receive6_2E[NB_LOCAL], *receive6_4E[NB_LOCAL], *receive6_6E[NB_LOCAL], *receive6_8E[NB_LOCAL];

double  *receive1_3E2[NB_LOCAL], *receive1_4E2[NB_LOCAL], *receive1_7E2[NB_LOCAL], *receive1_8E2[NB_LOCAL];
double  *receive2_1E2[NB_LOCAL], *receive2_2E2[NB_LOCAL], *receive2_3E2[NB_LOCAL], *receive2_4E2[NB_LOCAL];
double  *receive3_1E2[NB_LOCAL], *receive3_2E2[NB_LOCAL], *receive3_5E2[NB_LOCAL], *receive3_6E2[NB_LOCAL];
double  *receive4_5E2[NB_LOCAL], *receive4_6E2[NB_LOCAL], *receive4_7E2[NB_LOCAL], *receive4_8E2[NB_LOCAL];
double  *receive5_1E2[NB_LOCAL], *receive5_3E2[NB_LOCAL], *receive5_5E2[NB_LOCAL], *receive5_7E2[NB_LOCAL];
double  *receive6_2E2[NB_LOCAL], *receive6_4E2[NB_LOCAL], *receive6_6E2[NB_LOCAL], *receive6_8E2[NB_LOCAL];

double  *receive1_3E1[NB_LOCAL], *receive1_4E1[NB_LOCAL], *receive1_7E1[NB_LOCAL], *receive1_8E1[NB_LOCAL];
double  *receive2_1E1[NB_LOCAL], *receive2_2E1[NB_LOCAL], *receive2_3E1[NB_LOCAL], *receive2_4E1[NB_LOCAL];
double  *receive3_1E1[NB_LOCAL], *receive3_2E1[NB_LOCAL], *receive3_5E1[NB_LOCAL], *receive3_6E1[NB_LOCAL];
double  *receive4_5E1[NB_LOCAL], *receive4_6E1[NB_LOCAL], *receive4_7E1[NB_LOCAL], *receive4_8E1[NB_LOCAL];
double  *receive5_1E1[NB_LOCAL], *receive5_3E1[NB_LOCAL], *receive5_5E1[NB_LOCAL], *receive5_7E1[NB_LOCAL];
double  *receive6_2E1[NB_LOCAL], *receive6_4E1[NB_LOCAL], *receive6_6E1[NB_LOCAL], *receive6_8E1[NB_LOCAL];

double  *cornsend1[NB_LOCAL], *cornsend2[NB_LOCAL], *cornsend3[NB_LOCAL], *cornsend4[NB_LOCAL], *cornsend5[NB_LOCAL], *cornsend6[NB_LOCAL], *cornsend7[NB_LOCAL], *cornsend8[NB_LOCAL];
double *cornreceive1[NB_LOCAL], *cornreceive2[NB_LOCAL], *cornreceive3[NB_LOCAL], *cornreceive4[NB_LOCAL], *cornreceive5[NB_LOCAL], *cornreceive6[NB_LOCAL], *cornreceive7[NB_LOCAL], *cornreceive8[NB_LOCAL];

double *send_E3_corn1[NB_LOCAL], *send_E3_corn2[NB_LOCAL], *send_E3_corn3[NB_LOCAL], *send_E3_corn4[NB_LOCAL], *send_E2_corn5[NB_LOCAL], *send_E2_corn6[NB_LOCAL],
*send_E2_corn7[NB_LOCAL], *send_E2_corn8[NB_LOCAL], *send_E1_corn9[NB_LOCAL], *send_E1_corn10[NB_LOCAL], *send_E1_corn11[NB_LOCAL], *send_E1_corn12[NB_LOCAL];
double *receive_E3_corn1_1[NB_LOCAL], *receive_E3_corn2_1[NB_LOCAL], *receive_E3_corn3_1[NB_LOCAL], *receive_E3_corn4_1[NB_LOCAL], *receive_E2_corn5_1[NB_LOCAL], *receive_E2_corn6_1[NB_LOCAL],
*receive_E2_corn7_1[NB_LOCAL], *receive_E2_corn8_1[NB_LOCAL], *receive_E1_corn9_1[NB_LOCAL], *receive_E1_corn10_1[NB_LOCAL], *receive_E1_corn11_1[NB_LOCAL], *receive_E1_corn12_1[NB_LOCAL];
double *receive_E3_corn1_2[NB_LOCAL], *receive_E3_corn2_2[NB_LOCAL], *receive_E3_corn3_2[NB_LOCAL], *receive_E3_corn4_2[NB_LOCAL], *receive_E2_corn5_2[NB_LOCAL], *receive_E2_corn6_2[NB_LOCAL],
*receive_E2_corn7_2[NB_LOCAL], *receive_E2_corn8_2[NB_LOCAL], *receive_E1_corn9_2[NB_LOCAL], *receive_E1_corn10_2[NB_LOCAL], *receive_E1_corn11_2[NB_LOCAL], *receive_E1_corn12_2[NB_LOCAL];

double *tempreceive_E3_corn1_1[NB_LOCAL], *tempreceive_E3_corn2_1[NB_LOCAL], *tempreceive_E3_corn3_1[NB_LOCAL], *tempreceive_E3_corn4_1[NB_LOCAL], *tempreceive_E2_corn5_1[NB_LOCAL], *tempreceive_E2_corn6_1[NB_LOCAL],
*tempreceive_E2_corn7_1[NB_LOCAL], *tempreceive_E2_corn8_1[NB_LOCAL], *tempreceive_E1_corn9_1[NB_LOCAL], *tempreceive_E1_corn10_1[NB_LOCAL], *tempreceive_E1_corn11_1[NB_LOCAL], *tempreceive_E1_corn12_1[NB_LOCAL];
double *tempreceive_E3_corn1_2[NB_LOCAL], *tempreceive_E3_corn2_2[NB_LOCAL], *tempreceive_E3_corn3_2[NB_LOCAL], *tempreceive_E3_corn4_2[NB_LOCAL], *tempreceive_E2_corn5_2[NB_LOCAL], *tempreceive_E2_corn6_2[NB_LOCAL],
*tempreceive_E2_corn7_2[NB_LOCAL], *tempreceive_E2_corn8_2[NB_LOCAL], *tempreceive_E1_corn9_2[NB_LOCAL], *tempreceive_E1_corn10_2[NB_LOCAL], *tempreceive_E1_corn11_2[NB_LOCAL], *tempreceive_E1_corn12_2[NB_LOCAL];

double *receive_E3_corn1_12[NB_LOCAL], *receive_E3_corn2_12[NB_LOCAL], *receive_E3_corn3_12[NB_LOCAL], *receive_E3_corn4_12[NB_LOCAL], *receive_E2_corn5_12[NB_LOCAL], *receive_E2_corn6_12[NB_LOCAL],
*receive_E2_corn7_12[NB_LOCAL], *receive_E2_corn8_12[NB_LOCAL], *receive_E1_corn9_12[NB_LOCAL], *receive_E1_corn10_12[NB_LOCAL], *receive_E1_corn11_12[NB_LOCAL], *receive_E1_corn12_12[NB_LOCAL];
double *receive_E3_corn1_22[NB_LOCAL], *receive_E3_corn2_22[NB_LOCAL], *receive_E3_corn3_22[NB_LOCAL], *receive_E3_corn4_22[NB_LOCAL], *receive_E2_corn5_22[NB_LOCAL], *receive_E2_corn6_22[NB_LOCAL],
*receive_E2_corn7_22[NB_LOCAL], *receive_E2_corn8_22[NB_LOCAL], *receive_E1_corn9_22[NB_LOCAL], *receive_E1_corn10_22[NB_LOCAL], *receive_E1_corn11_22[NB_LOCAL], *receive_E1_corn12_22[NB_LOCAL];

double *receive_E3_corn1[NB_LOCAL], *receive_E3_corn2[NB_LOCAL], *receive_E3_corn3[NB_LOCAL], *receive_E3_corn4[NB_LOCAL];
double *receive_E2_corn5[NB_LOCAL], *receive_E2_corn6[NB_LOCAL], *receive_E2_corn7[NB_LOCAL], *receive_E2_corn8[NB_LOCAL];
double *receive_E1_corn9[NB_LOCAL], *receive_E1_corn10[NB_LOCAL], *receive_E1_corn11[NB_LOCAL], *receive_E1_corn12[NB_LOCAL];

double *tempreceive_E3_corn1[NB_LOCAL], *tempreceive_E3_corn2[NB_LOCAL], *tempreceive_E3_corn3[NB_LOCAL], *tempreceive_E3_corn4[NB_LOCAL];
double *tempreceive_E2_corn5[NB_LOCAL], *tempreceive_E2_corn6[NB_LOCAL], *tempreceive_E2_corn7[NB_LOCAL], *tempreceive_E2_corn8[NB_LOCAL];
double *tempreceive_E1_corn9[NB_LOCAL], *tempreceive_E1_corn10[NB_LOCAL], *tempreceive_E1_corn11[NB_LOCAL], *tempreceive_E1_corn12[NB_LOCAL];

/*gpu arrays decleration*/
double *NULL_POINTER[NB_LOCAL];
gpuStream_t commandQueue[NB_LOCAL];
gpuStream_t commandQueueGPU[NB_LOCAL];
gpuEvent_t boundevent[NB_LOCAL][600];
gpuEvent_t boundevent1[NB_LOCAL][100];
gpuEvent_t boundevent2[NB_LOCAL][100];
int fix_mem[NB_LOCAL];
int fix_mem2[NB_LOCAL];
int nr_workgroups[NB_LOCAL];
int nr_workgroups1[NB_LOCAL];
int nr_workgroups2[NB_LOCAL];
int nr_workgroups2_1[NB_LOCAL];
int nr_workgroups2_2[NB_LOCAL];
int nr_workgroups2_3[NB_LOCAL];
int nr_workgroups3[NB_LOCAL];
int nr_workgroups_special[NB_LOCAL];
int nr_workgroups_special1[NB_LOCAL];
int nr_workgroups_special2[NB_LOCAL];
int nr_workgroups_special3[NB_LOCAL];
int global_work_size[NB_LOCAL][1];
int global_work_size1[NB_LOCAL][1];
int global_work_size2[NB_LOCAL][1];
int global_work_size2_1[NB_LOCAL][1];
int global_work_size2_2[NB_LOCAL][1];
int global_work_size2_3[NB_LOCAL][1];
int global_work_size3[NB_LOCAL][1];
int global_work_size_bound[NB_LOCAL][1];
int global_work_offset[NB_LOCAL][1];
int global_work_size_special[NB_LOCAL][1];
int global_work_size_special1[NB_LOCAL][1];
int global_work_size_special2[NB_LOCAL][1];
int global_work_size_special3[NB_LOCAL][1];
int local_work_size[1];
double * Bufferconn[NB_LOCAL];
double * Buffergcov[NB_LOCAL];
double * Buffergcon[NB_LOCAL];
double * BufferMud[NB_LOCAL];
double * BufferMud_inv[NB_LOCAL];
double * Buffergdet[NB_LOCAL];
double * BufferF1_1[NB_LOCAL];
double * BufferF2_1[NB_LOCAL];
double * BufferF3_1[NB_LOCAL];
double * Bufferdq_1[NB_LOCAL];
double * BufferE_1[NB_LOCAL];
double * BufferV[NB_LOCAL];
double * BufferdU[NB_LOCAL];
double* BufferdU_RAD0[NB_LOCAL];
double* BufferdU_RAD1[NB_LOCAL];
double* BufferU_n[NB_LOCAL];
double* BufferU_0[NB_LOCAL];
double* BufferU_1[NB_LOCAL];
double * Bufferradius[NB_LOCAL];
double * Bufferstorage1[NB_LOCAL];
double * Bufferstorage2[NB_LOCAL];
double * Bufferstorage3[NB_LOCAL];
double * Bufferstorage4[NB_LOCAL];
double * Buffereta_avg[NB_LOCAL];
double * Bufferp_1[NB_LOCAL];
double * Bufferph_1[NB_LOCAL];
double * Bufferps_1[NB_LOCAL];
double * Bufferpsh_1[NB_LOCAL];
double * Bufferdtij1[NB_LOCAL];
double * Bufferdtij2[NB_LOCAL];
double * Bufferdtij3[NB_LOCAL];
int * Bufferpflag[NB_LOCAL];
int * Bufferpflag_CART[NB_LOCAL];
int * Bufferpflag_RAD[NB_LOCAL];
int * Bufferfailimage[NB_LOCAL];
double * BufferKatm[NB_LOCAL];
double * Buffersend1[NB_LOCAL];
double * Buffersend1_3[NB_LOCAL];
double * Buffersend1_4[NB_LOCAL];
double * Buffersend1_7[NB_LOCAL];
double * Buffersend1_8[NB_LOCAL];
double * Buffersend2[NB_LOCAL];
double * Buffersend2_1[NB_LOCAL];
double * Buffersend2_2[NB_LOCAL];
double * Buffersend2_3[NB_LOCAL];
double * Buffersend2_4[NB_LOCAL];
double * Buffersend3[NB_LOCAL];
double * Buffersend3_1[NB_LOCAL];
double * Buffersend3_2[NB_LOCAL];
double * Buffersend3_5[NB_LOCAL];
double * Buffersend3_6[NB_LOCAL];
double * Buffersend4[NB_LOCAL];
double * Buffersend4_5[NB_LOCAL];
double * Buffersend4_6[NB_LOCAL];
double * Buffersend4_7[NB_LOCAL];
double * Buffersend4_8[NB_LOCAL];
double * Buffersend5[NB_LOCAL];
double * Buffersend5_1[NB_LOCAL];
double * Buffersend5_3[NB_LOCAL];
double * Buffersend5_5[NB_LOCAL];
double * Buffersend5_7[NB_LOCAL];
double * Buffersend6[NB_LOCAL];
double * Buffersend6_2[NB_LOCAL];
double * Buffersend6_4[NB_LOCAL];
double * Buffersend6_6[NB_LOCAL];
double * Buffersend6_8[NB_LOCAL];
double * Bufferrec1[NB_LOCAL];
double * Bufferrec1_3[NB_LOCAL];
double * Bufferrec1_4[NB_LOCAL];
double * Bufferrec1_7[NB_LOCAL];
double * Bufferrec1_8[NB_LOCAL];
double * Bufferrec2[NB_LOCAL];
double * Bufferrec2_1[NB_LOCAL];
double * Bufferrec2_2[NB_LOCAL];
double * Bufferrec2_3[NB_LOCAL];
double * Bufferrec2_4[NB_LOCAL];
double * Bufferrec3[NB_LOCAL];
double * Bufferrec3_1[NB_LOCAL];
double * Bufferrec3_2[NB_LOCAL];
double * Bufferrec3_5[NB_LOCAL];
double * Bufferrec3_6[NB_LOCAL];
double * Bufferrec4[NB_LOCAL];
double * Bufferrec4_5[NB_LOCAL];
double * Bufferrec4_6[NB_LOCAL];
double * Bufferrec4_7[NB_LOCAL];
double * Bufferrec4_8[NB_LOCAL];
double * Bufferrec5[NB_LOCAL];
double * Bufferrec5_1[NB_LOCAL];
double * Bufferrec5_3[NB_LOCAL];
double * Bufferrec5_5[NB_LOCAL];
double * Bufferrec5_7[NB_LOCAL];
double * Bufferrec6[NB_LOCAL];
double * Bufferrec6_2[NB_LOCAL];
double * Bufferrec6_4[NB_LOCAL];
double * Bufferrec6_6[NB_LOCAL];
double * Bufferrec6_8[NB_LOCAL];

double * tempBufferrec1[NB_LOCAL];
double * tempBufferrec1_3[NB_LOCAL];
double * tempBufferrec1_4[NB_LOCAL];
double * tempBufferrec1_7[NB_LOCAL];
double * tempBufferrec1_8[NB_LOCAL];
double * tempBufferrec2[NB_LOCAL];
double * tempBufferrec2_1[NB_LOCAL];
double * tempBufferrec2_2[NB_LOCAL];
double * tempBufferrec2_3[NB_LOCAL];
double * tempBufferrec2_4[NB_LOCAL];
double * tempBufferrec3[NB_LOCAL];
double * tempBufferrec3_1[NB_LOCAL];
double * tempBufferrec3_2[NB_LOCAL];
double * tempBufferrec3_5[NB_LOCAL];
double * tempBufferrec3_6[NB_LOCAL];
double * tempBufferrec4[NB_LOCAL];
double * tempBufferrec4_5[NB_LOCAL];
double * tempBufferrec4_6[NB_LOCAL];
double * tempBufferrec4_7[NB_LOCAL];
double * tempBufferrec4_8[NB_LOCAL];
double * tempBufferrec5[NB_LOCAL];
double * tempBufferrec5_1[NB_LOCAL];
double * tempBufferrec5_3[NB_LOCAL];
double * tempBufferrec5_5[NB_LOCAL];
double * tempBufferrec5_7[NB_LOCAL];
double * tempBufferrec6[NB_LOCAL];
double * tempBufferrec6_2[NB_LOCAL];
double * tempBufferrec6_4[NB_LOCAL];
double * tempBufferrec6_6[NB_LOCAL];
double * tempBufferrec6_8[NB_LOCAL];

double * Buffersend1flux[NB_LOCAL];
double * Buffersend2flux[NB_LOCAL];
double * Buffersend3flux[NB_LOCAL];
double * Buffersend4flux[NB_LOCAL];
double * Buffersend5flux[NB_LOCAL];
double * Buffersend6flux[NB_LOCAL];
double * Bufferrec1flux[NB_LOCAL];
double * Bufferrec1_3flux[NB_LOCAL];
double * Bufferrec1_4flux[NB_LOCAL];
double * Bufferrec1_7flux[NB_LOCAL];
double * Bufferrec1_8flux[NB_LOCAL];
double * Bufferrec2flux[NB_LOCAL];
double * Bufferrec2_1flux[NB_LOCAL];
double * Bufferrec2_2flux[NB_LOCAL];
double * Bufferrec2_3flux[NB_LOCAL];
double * Bufferrec2_4flux[NB_LOCAL];
double * Bufferrec3flux[NB_LOCAL];
double * Bufferrec3_1flux[NB_LOCAL];
double * Bufferrec3_2flux[NB_LOCAL];
double * Bufferrec3_5flux[NB_LOCAL];
double * Bufferrec3_6flux[NB_LOCAL];
double * Bufferrec4flux[NB_LOCAL];
double * Bufferrec4_5flux[NB_LOCAL];
double * Bufferrec4_6flux[NB_LOCAL];
double * Bufferrec4_7flux[NB_LOCAL];
double * Bufferrec4_8flux[NB_LOCAL];
double * Bufferrec5flux[NB_LOCAL];
double * Bufferrec5_1flux[NB_LOCAL];
double * Bufferrec5_3flux[NB_LOCAL];
double * Bufferrec5_5flux[NB_LOCAL];
double * Bufferrec5_7flux[NB_LOCAL];
double * Bufferrec6flux[NB_LOCAL];
double * Bufferrec6_2flux[NB_LOCAL];
double * Bufferrec6_4flux[NB_LOCAL];
double * Bufferrec6_6flux[NB_LOCAL];
double * Bufferrec6_8flux[NB_LOCAL];

double * Bufferrec1flux1[NB_LOCAL];
double * Bufferrec2flux1[NB_LOCAL];
double * Bufferrec3flux1[NB_LOCAL];
double * Bufferrec4flux1[NB_LOCAL];
double * Bufferrec5flux1[NB_LOCAL];
double * Bufferrec6flux1[NB_LOCAL];


double * Bufferrec1_3flux1[NB_LOCAL];
double * Bufferrec1_4flux1[NB_LOCAL];
double * Bufferrec1_7flux1[NB_LOCAL];
double * Bufferrec1_8flux1[NB_LOCAL];
double * Bufferrec2_1flux1[NB_LOCAL];
double * Bufferrec2_2flux1[NB_LOCAL];
double * Bufferrec2_3flux1[NB_LOCAL];
double * Bufferrec2_4flux1[NB_LOCAL];
double * Bufferrec3_1flux1[NB_LOCAL];
double * Bufferrec3_2flux1[NB_LOCAL];
double * Bufferrec3_5flux1[NB_LOCAL];
double * Bufferrec3_6flux1[NB_LOCAL];
double * Bufferrec4_5flux1[NB_LOCAL];
double * Bufferrec4_6flux1[NB_LOCAL];
double * Bufferrec4_7flux1[NB_LOCAL];
double * Bufferrec4_8flux1[NB_LOCAL];
double * Bufferrec5_1flux1[NB_LOCAL];
double * Bufferrec5_3flux1[NB_LOCAL];
double * Bufferrec5_5flux1[NB_LOCAL];
double * Bufferrec5_7flux1[NB_LOCAL];
double * Bufferrec6_2flux1[NB_LOCAL];
double * Bufferrec6_4flux1[NB_LOCAL];
double * Bufferrec6_6flux1[NB_LOCAL];
double * Bufferrec6_8flux1[NB_LOCAL];

double * Bufferrec1_3flux2[NB_LOCAL];
double * Bufferrec1_4flux2[NB_LOCAL];
double * Bufferrec1_7flux2[NB_LOCAL];
double * Bufferrec1_8flux2[NB_LOCAL];
double * Bufferrec2_1flux2[NB_LOCAL];
double * Bufferrec2_2flux2[NB_LOCAL];
double * Bufferrec2_3flux2[NB_LOCAL];
double * Bufferrec2_4flux2[NB_LOCAL];
double * Bufferrec3_1flux2[NB_LOCAL];
double * Bufferrec3_2flux2[NB_LOCAL];
double * Bufferrec3_5flux2[NB_LOCAL];
double * Bufferrec3_6flux2[NB_LOCAL];
double * Bufferrec4_5flux2[NB_LOCAL];
double * Bufferrec4_6flux2[NB_LOCAL];
double * Bufferrec4_7flux2[NB_LOCAL];
double * Bufferrec4_8flux2[NB_LOCAL];
double * Bufferrec5_1flux2[NB_LOCAL];
double * Bufferrec5_3flux2[NB_LOCAL];
double * Bufferrec5_5flux2[NB_LOCAL];
double * Bufferrec5_7flux2[NB_LOCAL];
double * Bufferrec6_2flux2[NB_LOCAL];
double * Bufferrec6_4flux2[NB_LOCAL];
double * Bufferrec6_6flux2[NB_LOCAL];
double * Bufferrec6_8flux2[NB_LOCAL];

double * Buffersend1fine[NB_LOCAL];
double * Buffersend2fine[NB_LOCAL];
double * Buffersend3fine[NB_LOCAL];
double * Buffersend4fine[NB_LOCAL];
double * Buffersend5fine[NB_LOCAL];
double * Buffersend6fine[NB_LOCAL];
double * Bufferrec1fine[NB_LOCAL];
double * Bufferrec2fine[NB_LOCAL];
double * Bufferrec3fine[NB_LOCAL];
double * Bufferrec4fine[NB_LOCAL];
double * Bufferrec5fine[NB_LOCAL];
double * Bufferrec6fine[NB_LOCAL];
double * Bufferrec1_3fine[NB_LOCAL];
double * Bufferrec1_4fine[NB_LOCAL];
double * Bufferrec1_7fine[NB_LOCAL];
double * Bufferrec1_8fine[NB_LOCAL];
double * Bufferrec2_1fine[NB_LOCAL];
double * Bufferrec2_2fine[NB_LOCAL];
double * Bufferrec2_3fine[NB_LOCAL];
double * Bufferrec2_4fine[NB_LOCAL];
double * Bufferrec3_1fine[NB_LOCAL];
double * Bufferrec3_2fine[NB_LOCAL];
double * Bufferrec3_5fine[NB_LOCAL];
double * Bufferrec3_6fine[NB_LOCAL];
double * Bufferrec4_5fine[NB_LOCAL];
double * Bufferrec4_6fine[NB_LOCAL];
double * Bufferrec4_7fine[NB_LOCAL];
double * Bufferrec4_8fine[NB_LOCAL];
double * Bufferrec5_1fine[NB_LOCAL];
double * Bufferrec5_3fine[NB_LOCAL];
double * Bufferrec5_5fine[NB_LOCAL];
double * Bufferrec5_7fine[NB_LOCAL];
double * Bufferrec6_2fine[NB_LOCAL];
double * Bufferrec6_4fine[NB_LOCAL];
double * Bufferrec6_6fine[NB_LOCAL];
double * Bufferrec6_8fine[NB_LOCAL];

double * Buffersend1E[NB_LOCAL];
double * Buffersend2E[NB_LOCAL];
double * Buffersend3E[NB_LOCAL];
double * Buffersend4E[NB_LOCAL];
double * Buffersend5E[NB_LOCAL];
double * Buffersend6E[NB_LOCAL];
double * Bufferrec1E[NB_LOCAL];
double * Bufferrec2E[NB_LOCAL];
double * Bufferrec3E[NB_LOCAL];
double * Bufferrec4E[NB_LOCAL];
double * Bufferrec5E[NB_LOCAL];
double * Bufferrec6E[NB_LOCAL];
double * Bufferrec1E1[NB_LOCAL];
double * Bufferrec2E1[NB_LOCAL];
double * Bufferrec3E1[NB_LOCAL];
double * Bufferrec4E1[NB_LOCAL];
double * Bufferrec5E1[NB_LOCAL];
double * Bufferrec6E1[NB_LOCAL];
double * Bufferrec1_3E[NB_LOCAL];
double * Bufferrec1_4E[NB_LOCAL];
double * Bufferrec1_7E[NB_LOCAL];
double * Bufferrec1_8E[NB_LOCAL];
double * Bufferrec2_1E[NB_LOCAL];
double * Bufferrec2_2E[NB_LOCAL];
double * Bufferrec2_3E[NB_LOCAL];
double * Bufferrec2_4E[NB_LOCAL];
double * Bufferrec3_1E[NB_LOCAL];
double * Bufferrec3_2E[NB_LOCAL];
double * Bufferrec3_5E[NB_LOCAL];
double * Bufferrec3_6E[NB_LOCAL];
double * Bufferrec4_5E[NB_LOCAL];
double * Bufferrec4_6E[NB_LOCAL];
double * Bufferrec4_7E[NB_LOCAL];
double * Bufferrec4_8E[NB_LOCAL];
double * Bufferrec5_1E[NB_LOCAL];
double * Bufferrec5_3E[NB_LOCAL];
double * Bufferrec5_5E[NB_LOCAL];
double * Bufferrec5_7E[NB_LOCAL];
double * Bufferrec6_2E[NB_LOCAL];
double * Bufferrec6_4E[NB_LOCAL];
double * Bufferrec6_6E[NB_LOCAL];
double * Bufferrec6_8E[NB_LOCAL];

double * Bufferrec1_3E1[NB_LOCAL];
double * Bufferrec1_4E1[NB_LOCAL];
double * Bufferrec1_7E1[NB_LOCAL];
double * Bufferrec1_8E1[NB_LOCAL];
double * Bufferrec2_1E1[NB_LOCAL];
double * Bufferrec2_2E1[NB_LOCAL];
double * Bufferrec2_3E1[NB_LOCAL];
double * Bufferrec2_4E1[NB_LOCAL];
double * Bufferrec3_1E1[NB_LOCAL];
double * Bufferrec3_2E1[NB_LOCAL];
double * Bufferrec3_5E1[NB_LOCAL];
double * Bufferrec3_6E1[NB_LOCAL];
double * Bufferrec4_5E1[NB_LOCAL];
double * Bufferrec4_6E1[NB_LOCAL];
double * Bufferrec4_7E1[NB_LOCAL];
double * Bufferrec4_8E1[NB_LOCAL];
double * Bufferrec5_1E1[NB_LOCAL];
double * Bufferrec5_3E1[NB_LOCAL];
double * Bufferrec5_5E1[NB_LOCAL];
double * Bufferrec5_7E1[NB_LOCAL];
double * Bufferrec6_2E1[NB_LOCAL];
double * Bufferrec6_4E1[NB_LOCAL];
double * Bufferrec6_6E1[NB_LOCAL];
double * Bufferrec6_8E1[NB_LOCAL];

double * Bufferrec1_3E2[NB_LOCAL];
double * Bufferrec1_4E2[NB_LOCAL];
double * Bufferrec1_7E2[NB_LOCAL];
double * Bufferrec1_8E2[NB_LOCAL];
double * Bufferrec2_1E2[NB_LOCAL];
double * Bufferrec2_2E2[NB_LOCAL];
double * Bufferrec2_3E2[NB_LOCAL];
double * Bufferrec2_4E2[NB_LOCAL];
double * Bufferrec3_1E2[NB_LOCAL];
double * Bufferrec3_2E2[NB_LOCAL];
double * Bufferrec3_5E2[NB_LOCAL];
double * Bufferrec3_6E2[NB_LOCAL];
double * Bufferrec4_5E2[NB_LOCAL];
double * Bufferrec4_6E2[NB_LOCAL];
double * Bufferrec4_7E2[NB_LOCAL];
double * Bufferrec4_8E2[NB_LOCAL];
double * Bufferrec5_1E2[NB_LOCAL];
double * Bufferrec5_3E2[NB_LOCAL];
double * Bufferrec5_5E2[NB_LOCAL];
double * Bufferrec5_7E2[NB_LOCAL];
double * Bufferrec6_2E2[NB_LOCAL];
double * Bufferrec6_4E2[NB_LOCAL];
double * Bufferrec6_6E2[NB_LOCAL];
double * Bufferrec6_8E2[NB_LOCAL];

double * BuffersendE1corn9[NB_LOCAL];
double * BuffersendE1corn10[NB_LOCAL];
double * BuffersendE1corn11[NB_LOCAL];
double * BuffersendE1corn12[NB_LOCAL];
double * BuffersendE2corn5[NB_LOCAL];
double * BuffersendE2corn6[NB_LOCAL];
double * BuffersendE2corn7[NB_LOCAL];
double * BuffersendE2corn8[NB_LOCAL];
double * BuffersendE3corn1[NB_LOCAL];
double * BuffersendE3corn2[NB_LOCAL];
double * BuffersendE3corn3[NB_LOCAL];
double * BuffersendE3corn4[NB_LOCAL];
double * BufferrecE1corn9[NB_LOCAL];
double * BufferrecE1corn10[NB_LOCAL];
double * BufferrecE1corn11[NB_LOCAL];
double * BufferrecE1corn12[NB_LOCAL];
double * BufferrecE2corn5[NB_LOCAL];
double * BufferrecE2corn6[NB_LOCAL];
double * BufferrecE2corn7[NB_LOCAL];
double * BufferrecE2corn8[NB_LOCAL];
double * BufferrecE3corn1[NB_LOCAL];
double * BufferrecE3corn2[NB_LOCAL];
double * BufferrecE3corn3[NB_LOCAL];
double * BufferrecE3corn4[NB_LOCAL];

double * tempBufferrecE1corn9[NB_LOCAL];
double * tempBufferrecE1corn10[NB_LOCAL];
double * tempBufferrecE1corn11[NB_LOCAL];
double * tempBufferrecE1corn12[NB_LOCAL];
double * tempBufferrecE2corn5[NB_LOCAL];
double * tempBufferrecE2corn6[NB_LOCAL];
double * tempBufferrecE2corn7[NB_LOCAL];
double * tempBufferrecE2corn8[NB_LOCAL];
double * tempBufferrecE3corn1[NB_LOCAL];
double * tempBufferrecE3corn2[NB_LOCAL];
double * tempBufferrecE3corn3[NB_LOCAL];
double * tempBufferrecE3corn4[NB_LOCAL];

double * BufferrecE1corn9_3[NB_LOCAL];
double * BufferrecE1corn9_7[NB_LOCAL];
double * BufferrecE1corn10_1[NB_LOCAL];
double * BufferrecE1corn10_5[NB_LOCAL];
double * BufferrecE1corn11_2[NB_LOCAL];
double * BufferrecE1corn11_6[NB_LOCAL];
double * BufferrecE1corn12_4[NB_LOCAL];
double * BufferrecE1corn12_8[NB_LOCAL];
double * BufferrecE2corn5_2[NB_LOCAL];
double * BufferrecE2corn5_4[NB_LOCAL];
double * BufferrecE2corn6_1[NB_LOCAL];
double * BufferrecE2corn6_3[NB_LOCAL];
double * BufferrecE2corn7_5[NB_LOCAL];
double * BufferrecE2corn7_7[NB_LOCAL];
double * BufferrecE2corn8_6[NB_LOCAL];
double * BufferrecE2corn8_8[NB_LOCAL];
double * BufferrecE3corn1_3[NB_LOCAL];
double * BufferrecE3corn1_4[NB_LOCAL];
double * BufferrecE3corn2_1[NB_LOCAL];
double * BufferrecE3corn2_2[NB_LOCAL];
double * BufferrecE3corn3_5[NB_LOCAL];
double * BufferrecE3corn3_6[NB_LOCAL];
double * BufferrecE3corn4_7[NB_LOCAL];
double * BufferrecE3corn4_8[NB_LOCAL];

double * tempBufferrecE1corn9_3[NB_LOCAL];
double * tempBufferrecE1corn9_7[NB_LOCAL];
double * tempBufferrecE1corn10_1[NB_LOCAL];
double * tempBufferrecE1corn10_5[NB_LOCAL];
double * tempBufferrecE1corn11_2[NB_LOCAL];
double * tempBufferrecE1corn11_6[NB_LOCAL];
double * tempBufferrecE1corn12_4[NB_LOCAL];
double * tempBufferrecE1corn12_8[NB_LOCAL];
double * tempBufferrecE2corn5_2[NB_LOCAL];
double * tempBufferrecE2corn5_4[NB_LOCAL];
double * tempBufferrecE2corn6_1[NB_LOCAL];
double * tempBufferrecE2corn6_3[NB_LOCAL];
double * tempBufferrecE2corn7_5[NB_LOCAL];
double * tempBufferrecE2corn7_7[NB_LOCAL];
double * tempBufferrecE2corn8_6[NB_LOCAL];
double * tempBufferrecE2corn8_8[NB_LOCAL];
double * tempBufferrecE3corn1_3[NB_LOCAL];
double * tempBufferrecE3corn1_4[NB_LOCAL];
double * tempBufferrecE3corn2_1[NB_LOCAL];
double * tempBufferrecE3corn2_2[NB_LOCAL];
double * tempBufferrecE3corn3_5[NB_LOCAL];
double * tempBufferrecE3corn3_6[NB_LOCAL];
double * tempBufferrecE3corn4_7[NB_LOCAL];
double * tempBufferrecE3corn4_8[NB_LOCAL];

double * BufferrecE1corn9_32[NB_LOCAL];
double * BufferrecE1corn9_72[NB_LOCAL];
double * BufferrecE1corn10_12[NB_LOCAL];
double * BufferrecE1corn10_52[NB_LOCAL];
double * BufferrecE1corn11_22[NB_LOCAL];
double * BufferrecE1corn11_62[NB_LOCAL];
double * BufferrecE1corn12_42[NB_LOCAL];
double * BufferrecE1corn12_82[NB_LOCAL];
double * BufferrecE2corn5_22[NB_LOCAL];
double * BufferrecE2corn5_42[NB_LOCAL];
double * BufferrecE2corn6_12[NB_LOCAL];
double * BufferrecE2corn6_32[NB_LOCAL];
double * BufferrecE2corn7_52[NB_LOCAL];
double * BufferrecE2corn7_72[NB_LOCAL];
double * BufferrecE2corn8_62[NB_LOCAL];
double * BufferrecE2corn8_82[NB_LOCAL];
double * BufferrecE3corn1_32[NB_LOCAL];
double * BufferrecE3corn1_42[NB_LOCAL];
double * BufferrecE3corn2_12[NB_LOCAL];
double * BufferrecE3corn2_22[NB_LOCAL];
double * BufferrecE3corn3_52[NB_LOCAL];
double * BufferrecE3corn3_62[NB_LOCAL];
double * BufferrecE3corn4_72[NB_LOCAL];
double * BufferrecE3corn4_82[NB_LOCAL];

#if (DOHELM)
double * eos_table[1];
double * GPU_eos_table[1];
#endif

/*************************************************************************
GLOBAL VARIABLES SECTION
*************************************************************************/
/* physics parameters */
double a;
double gam;

/* numerical parameters */
double Rin, Rout, R0, fractheta, x1br, rbr, npow2, cpow2, x1max;
double cour;
double dV, dx[NB_LOCAL][NPR], startx[NPR];
double dt, bdt[NB_LOCAL][4];
int NODE_global[NB];
double t, tf;
int nstep;
double sourceflag, period_max;
double rmax;
double ndt, ndt1, ndt2, ndt3;
int numtasks, rank, local_rank, rc;
int prestep_half[NB_LOCAL], prestep_full[NB_LOCAL];
int max_levels, numdevices;
int reduce_timestep;
int nthreads;
int gpu, gpu_offset;
int status;
double y_max;
double mass_density_scale_cpu, magnetic_density_scale_cpu, mdot_cpu, t_mdot;

/* output parameters */
double DTd;
double DTd_reduced;
double DTl;
double DTi;
int    DTr;
double tref;
int    dump_cnt, dump_cnt_reduced;
int    image_cnt;
int    rdump_cnt;

/* global flags */
int failed;
int lim;
double defcon;

/* set global variables that indicate current local metric, etc. */
int icurr, jcurr, pcurr;

/*Timing/benchmarking decleration*/
clock_t begin1, end1, begin2, end2;
double time_spent3;

/*Parallel write*/
double *dump_buffer;
double(*restrict dxdxp_z[NB_LOCAL])[NDIM][NDIM];
double(*restrict dxpdx_z[NB_LOCAL])[NDIM][NDIM];
float *array[NB_LOCAL], *array_reduced[NB_LOCAL], *array_diag[NB_LOCAL];
int *array_gdumpgrid, *array_rdumpgrid;
double *array_rdump[NB_LOCAL], *array_gdump1[NB_LOCAL], *array_gdump2[NB_LOCAL], *array_gdump1_reduced[NB_LOCAL], *array_gdump2_reduced[NB_LOCAL];
int first_dump, first_dump_reduced, first_rdump, first_gdump, restart_number;
FILE *fparam_dump, *fparam_dump_reduced, *fparam_restart;
int read_M1, read_2T, read_Pnum, read_Res;
int BS1_read, BS2_read, BS3_read;

/*AMR parameters*/
int(*block)[NV];
int(*block_read)[10];
int *lin_coord[N_LEVELS];
int *lin_coord_RM[N_LEVELS];
double ref_val[MY_MAX(NB, 40000)];
double bench_time[NB];
int n_ord[NB_LOCAL], nl[NB], n_old[NB], n_ord_total[NB], n_ord_RM[NB_LOCAL], n_ord_total_RM[NB],(*n_ord_node)[NB_LOCAL];
int n_active, *n_active_node, n_active_total, n_max;
int mem_spot[NB_LOCAL], mem_spot_gpu[NB_LOCAL], mem_spot_gpu_bound[NB_LOCAL];
int count_node[1], count_gpu[N_GPU];
int N1_GPU_offset[NB];
int N2_GPU_offset[NB];
int N3_GPU_offset[NB];
int max_blocks;
float jet_angle1[NB_1 * BS_1], jet_angle2[NB_1 * BS_1], coccoon_angle1[NB_1 * BS_1], coccoon_angle2[NB_1 * BS_1], r_head1, r_head2;

//MPI Variables
MPI_Request req[NB], boundreqs[NB_LOCAL][600];
MPI_Status Statbound[NB_LOCAL][600];
MPI_Comm  mpi_cartcomm, mpi_self;
MPI_Comm row_comm[8];
MPI_File fdump[2000], fdump_reduced[2000], fdumpdiag[2000], rdump[NB_LOCAL], gdump[NB_LOCAL], gdump_reduced[NB_LOCAL], grid_dump[1], grid_restart[1];
MPI_Request req_block[NB_LOCAL][1], req_block_reduced[NB_LOCAL][1], req_block_rdump[NB_LOCAL][1], req_blockdiag[NB_LOCAL][1], req_gdump1[NB_LOCAL][1], req_gdump2[NB_LOCAL][1], req_gdump1_reduced[NB_LOCAL][1], req_gdump2_reduced[NB_LOCAL][1], req_gdumpgrid[1], req_rdumpgrid[1];
MPI_Request request_timelevel[NB];
MPI_Request req_local1[N_LEVELS_3D][NB_1*NB_3 * 64], req_local2[N_LEVELS_3D][NB_1*NB_3 * 64];
int send_tag1[N_LEVELS_3D][MY_MAX(NB, 60000)], send_tag2[N_LEVELS_3D][MY_MAX(NB, 60000)];
