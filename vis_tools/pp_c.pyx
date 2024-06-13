from numpy cimport ndarray
import numpy as np
cimport numpy as np
cimport cython
import os
import ctypes
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

cdef extern from "math.h":
    float sin(float x)
    float cos(float x)
cdef extern void calc_cart_c(float r, float h, float ph, float* dxdr)
cdef extern void get_state_c(float *uu, float *ud, float *bu, float *bd, float *gcov)
cdef extern void lower_c(float *uu, float *ud, float *gcov)
cdef extern void Tcalcuu_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tuu)
cdef extern void Tcalcud_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tud)
cdef extern void kernel_calc_prec_disk(int bs1, int bs2, int bs3, int nb, int axisym, int avg, float *r, float *h, float *ph, float *rho, float *ug, float *uu, float *B, float gam, float* gcov, float* gcon, float* gdet, float* dxdxp,float *Su_disk, float *L_disk,float *Su_corona, float *L_corona,float *Su_disk_avg, float *L_disk_avg,float *Su_corona_avg, float *L_corona_avg)
cdef extern void kernel_mdot(int bs1, int bs2, int bs3, int nb, int a_ndim, int b_ndim, float *a, float *b, float *c)
cdef extern void kernel_misc_calc(int bs1, int bs2, int bs3, int nb, int axisym, float *uu, float *B, float *E, float *bu, float *eu, float *gcov, float *bsq,float *esq, int calc_bu, int calc_eu, int calc_bsq, int calc_esq)
cdef extern void kernel_rdump_new(int flag,int RAD_M1, int RESISTIVE,int TWO_t, int P_NUM,  char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon,axisym)
cdef extern void kernel_rgdump_new(int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet)
cdef extern void kernel_rdump_write(int flag,int RAD_M1,int RESISTIVE, int TWO_t, int P_NUM, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon,axisym)
cdef extern void kernel_rgdump_write(int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet)
cdef extern void kernel_griddata3D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1,int nb2,int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3)
cdef extern void kernel_griddata2D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1,int nb2,int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3)
cdef extern void kernel_rgdump_griddata(int flag,int interpolate, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet,int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3, int nb1, int nb2, int nb3, int REF_1, int REF_2, int REF_3, float startx1, float startx2, float startx3, float _dx1, float _dx2, float _dx3, int export_raytracing, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max);
cdef extern void kernel_rdump_griddata(int flag, int interpolate,int RAD_M1,int RESISTIVE,int TWO_t, int P_NUM,  char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon, int axisym,int* n_ord,int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3,int nb1, int nb2, int nb3, int REF_1, int REF_2, int REF_3, int export_raytracing,float DISK_THICKNESS, float a, float gam, float* Rdot, float* bsq, float* r, float startx1, float startx2, float startx3, float _dx1, float _dx2, float _dx3, float* x1, float* x2, float* x3, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max);
cdef extern void invert_4x4(float *a, float *b)
cdef extern void kernel_invert_4x4(float *A, float *B, int nb, int bs1, int bs2, int bs3)

def pointwise_invert_4x4(np.ndarray[np.float32_t, ndim=6, mode="c"]  A, nb, bs1, bs2, bs3):
    '''

    Description:

    Assume input is array with form:
    A[4,4,nb,bs1,bs2,bs3]
    Then, loop through nb, bs1,bs2,bs3 and invert the associated 4x4 matric
    Fill B[4,4,nb,bs1,bs2,bs3] with nb*bs1*bs2*bs3 inverted matrices and return B
    '''
    cdef np.ndarray[np.float32_t, ndim=6, mode="c"] B = np.zeros((4,4,nb,bs1,bs2,bs3), dtype=np.float32, order='C')
    kernel_invert_4x4(&A[0,0,0,0,0,0],&B[0,0,0,0,0,0], nb, bs1, bs2, bs3)
    return B    

def rgdump_new(flag, dir, axisym, np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord,  f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  x1,np.ndarray[np.float32_t, ndim=4, mode="c"]  x2, np.ndarray[np.float32_t, ndim=4, mode="c"]  x3,np.ndarray[np.float32_t, ndim=4, mode="c"]  r,np.ndarray[np.float32_t, ndim=4, mode="c"]  h, np.ndarray[np.float32_t, ndim=4, mode="c"]  ph,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon,np.ndarray[np.float32_t, ndim=6, mode="c"]  dxdxp,np.ndarray[np.float32_t, ndim=4, mode="c"]  gdet):
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    py_byte_string = dir.encode('UTF-8')
    cdef char* c_string = py_byte_string
    kernel_rgdump_new(flag,c_string, axisym, &n_ord_c[0], f1, f2, f3, nb, bs1, bs2, bs3, &x1[0,0,0,0],&x2[0,0,0,0],&x3[0,0,0,0],&r[0,0,0,0],&h[0,0,0,0],&ph[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],&dxdxp[0,0,0,0,0,0],&gdet[0,0,0,0])

def rgdump_griddata(flag, interpolate, dir, axisym, np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord,  f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  x1,np.ndarray[np.float32_t, ndim=4, mode="c"]  x2, np.ndarray[np.float32_t, ndim=4, mode="c"]  x3,np.ndarray[np.float32_t, ndim=4, mode="c"]  r,np.ndarray[np.float32_t, ndim=4, mode="c"]  h, np.ndarray[np.float32_t, ndim=4, mode="c"]  ph,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon,np.ndarray[np.float32_t, ndim=6, mode="c"]  dxdxp,np.ndarray[np.float32_t, ndim=4, mode="c"]  gdet,np.ndarray[np.int32_t, ndim=2, mode="c"] block, nb1,nb2,nb3, REF_1, REF_2, REF_3, ACTIVE_1, ACTIVE2, ACTIVE_3, startx1, startx2, fstartx3, _dx1, _dx2, _dx3, export_raytracing, i_min, i_max, j_min, j_max, z_min, z_max):
    py_byte_string = dir.encode('UTF-8')
    cdef char* c_string = py_byte_string
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL3_c
    AMR_LEVEL1_c = np.ascontiguousarray(block[:,AMR_LEVEL1], dtype=ctypes.c_int)
    AMR_LEVEL2_c = np.ascontiguousarray(block[:,AMR_LEVEL2], dtype=ctypes.c_int)
    AMR_LEVEL3_c = np.ascontiguousarray(block[:,AMR_LEVEL3], dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD3_c
    AMR_COORD1_c = np.ascontiguousarray(block[:,AMR_COORD1], dtype=ctypes.c_int)
    AMR_COORD2_c = np.ascontiguousarray(block[:,AMR_COORD2], dtype=ctypes.c_int)
    AMR_COORD3_c = np.ascontiguousarray(block[:,AMR_COORD3], dtype=ctypes.c_int)
    kernel_rgdump_griddata(flag, interpolate, c_string, axisym, &n_ord_c[0], f1, f2, f3, nb, bs1, bs2, bs3, &x1[0,0,0,0],&x2[0,0,0,0],&x3[0,0,0,0],&r[0,0,0,0],&h[0,0,0,0],&ph[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],&dxdxp[0,0,0,0,0,0],&gdet[0,0,0,0], ACTIVE_1, ACTIVE2, ACTIVE_3, &AMR_LEVEL1_c[0], &AMR_LEVEL2_c[0], &AMR_LEVEL3_c[0], &AMR_COORD1_c[0], &AMR_COORD2_c[0], &AMR_COORD3_c[0], nb1, nb2, nb3, REF_1, REF_2, REF_3, startx1, startx2, fstartx3, _dx1, _dx2, _dx3, export_raytracing, i_min, i_max, j_min, j_max, z_min, z_max)

def rdump_new(flag,RAD_M1, RESISTIVE, TWO_T, P_NUM, dir, dump, n_active_total, f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  rho,np.ndarray[np.float32_t, ndim=4, mode="c"]  ug, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu, np.ndarray[np.float32_t, ndim=5, mode="c"]  B,np.ndarray[np.float32_t, ndim=5, mode="c"]  E,np.ndarray[np.float32_t, ndim=4, mode="c"]  E_rad, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu_rad,np.ndarray[np.float32_t, ndim=4, mode="c"]  TE,np.ndarray[np.float32_t, ndim=4, mode="c"]  TI,np.ndarray[np.float32_t, ndim=4, mode="c"]  photon_number, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon, axisym):
    py_byte_string = dir.encode('UTF-8')
    cdef char* c_string = py_byte_string
    kernel_rdump_new(flag, RAD_M1, RESISTIVE,TWO_T, P_NUM, c_string, dump, n_active_total, f1, f2, f3, nb, bs1, bs2, bs3, &rho[0,0,0,0], &ug[0,0,0,0], &uu[0,0,0,0,0], &B[0,0,0,0,0],&E[0,0,0,0,0],&E_rad[0,0,0,0],&uu_rad[0,0,0,0,0],&TE[0,0,0,0],&TI[0,0,0,0],&photon_number[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],axisym)

def rdump_griddata(flag, interpolate, RAD_M1, RESISTIVE, TWO_T, P_NUM, dir, dump, n_active_total, f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  rho,np.ndarray[np.float32_t, ndim=4, mode="c"]  ug, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu, np.ndarray[np.float32_t, ndim=5, mode="c"]  B,np.ndarray[np.float32_t, ndim=5, mode="c"]  E,np.ndarray[np.float32_t, ndim=4, mode="c"]  E_rad, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu_rad, np.ndarray[np.float32_t, ndim=4, mode="c"]  TE,np.ndarray[np.float32_t, ndim=4, mode="c"]  TI,np.ndarray[np.float32_t, ndim=4, mode="c"]  photon_number,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon, axisym,np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord,np.ndarray[np.int32_t, ndim=2, mode="c"] block, nb1,nb2,nb3,REF_1, REF_2, REF_3, ACTIVE_1, ACTIVE2, ACTIVE_3, export_raytracing,DISK_THICKNESS,a,gam, np.ndarray[np.float32_t, ndim=4, mode="c"] Rdot,np.ndarray[np.float32_t, ndim=4, mode="c"] bsq, np.ndarray[np.float32_t, ndim=4, mode="c"] r, startx1, startx2, fstartx3, _dx1, _dx2, _dx3, np.ndarray[np.float32_t, ndim=4, mode="c"]  x1,np.ndarray[np.float32_t, ndim=4, mode="c"]  x2, np.ndarray[np.float32_t, ndim=4, mode="c"]  x3, i_min, i_max, j_min, j_max, z_min, z_max):
    py_byte_string = dir.encode('UTF-8')
    cdef char* c_string = py_byte_string
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL3_c
    AMR_LEVEL1_c = np.ascontiguousarray(block[:,AMR_LEVEL1], dtype=ctypes.c_int)
    AMR_LEVEL2_c = np.ascontiguousarray(block[:,AMR_LEVEL2], dtype=ctypes.c_int)
    AMR_LEVEL3_c = np.ascontiguousarray(block[:,AMR_LEVEL3], dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD3_c
    AMR_COORD1_c = np.ascontiguousarray(block[:,AMR_COORD1], dtype=ctypes.c_int)
    AMR_COORD2_c = np.ascontiguousarray(block[:,AMR_COORD2], dtype=ctypes.c_int)
    AMR_COORD3_c = np.ascontiguousarray(block[:,AMR_COORD3], dtype=ctypes.c_int)
    kernel_rdump_griddata(flag, interpolate, RAD_M1, RESISTIVE, TWO_T, P_NUM,c_string, dump, n_active_total, f1, f2, f3, nb, bs1, bs2, bs3, &rho[0,0,0,0], &ug[0,0,0,0], &uu[0,0,0,0,0], &B[0,0,0,0,0], &E[0,0,0,0,0],&E_rad[0,0,0,0],&uu_rad[0,0,0,0,0],&TE[0,0,0,0],&TI[0,0,0,0],&photon_number[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],axisym, &n_ord_c[0],ACTIVE_1,ACTIVE2, ACTIVE_3, &AMR_LEVEL1_c[0], &AMR_LEVEL2_c[0], &AMR_LEVEL3_c[0], &AMR_COORD1_c[0], &AMR_COORD2_c[0], &AMR_COORD3_c[0], nb1, nb2, nb3, REF_1, REF_2, REF_3,export_raytracing, DISK_THICKNESS,a,gam, &Rdot[0,0,0,0],&bsq[0,0,0,0], &r[0,0,0,0], startx1, startx2, fstartx3, _dx1, _dx2, _dx3, &x1[0,0,0,0],&x2[0,0,0,0],&x3[0,0,0,0], i_min, i_max, j_min, j_max, z_min, z_max)

def griddata3D(n_active_total, bs1,bs2,bs3,nb1,nb2,nb3, np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord, np.ndarray[np.int32_t, ndim=2, mode="c"] block, np.ndarray[np.float32_t, ndim=4, mode="c"]  input,np.ndarray[np.float32_t, ndim=4, mode="c"] output, ACTIVE_1, ACTIVE2, ACTIVE_3):
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL3_c
    AMR_LEVEL1_c = np.ascontiguousarray(block[:,AMR_LEVEL1], dtype=ctypes.c_int)
    AMR_LEVEL2_c = np.ascontiguousarray(block[:,AMR_LEVEL2], dtype=ctypes.c_int)
    AMR_LEVEL3_c = np.ascontiguousarray(block[:,AMR_LEVEL3], dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD3_c
    AMR_COORD1_c = np.ascontiguousarray(block[:,AMR_COORD1], dtype=ctypes.c_int)
    AMR_COORD2_c = np.ascontiguousarray(block[:,AMR_COORD2], dtype=ctypes.c_int)
    AMR_COORD3_c = np.ascontiguousarray(block[:,AMR_COORD3], dtype=ctypes.c_int)

    kernel_griddata3D_new(n_active_total, bs1, bs2, bs3, nb1, nb2, nb3, &n_ord_c[0],&input[0,0,0,0], &output[0,0,0,0], ACTIVE_1, ACTIVE2, ACTIVE_3, &AMR_LEVEL1_c[0], &AMR_LEVEL2_c[0], &AMR_LEVEL3_c[0], &AMR_COORD1_c[0], &AMR_COORD2_c[0], &AMR_COORD3_c[0])

def griddata2D(n_active_total, bs1,bs2,bs3,nb1,nb2,nb3, np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord, np.ndarray[np.int32_t, ndim=2, mode="c"] block, np.ndarray[np.float32_t, ndim=4, mode="c"]  input,np.ndarray[np.float32_t, ndim=4, mode="c"] output, ACTIVE_1, ACTIVE2, ACTIVE_3):
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_LEVEL3_c
    AMR_LEVEL1_c = np.ascontiguousarray(block[:,AMR_LEVEL1], dtype=ctypes.c_int)
    AMR_LEVEL2_c = np.ascontiguousarray(block[:,AMR_LEVEL2], dtype=ctypes.c_int)
    AMR_LEVEL3_c = np.ascontiguousarray(block[:,AMR_LEVEL3], dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD1_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD2_c
    cdef np.ndarray[int, ndim=1, mode="c"] AMR_COORD3_c
    AMR_COORD1_c = np.ascontiguousarray(block[:,AMR_COORD1], dtype=ctypes.c_int)
    AMR_COORD2_c = np.ascontiguousarray(block[:,AMR_COORD2], dtype=ctypes.c_int)
    AMR_COORD3_c = np.ascontiguousarray(block[:,AMR_COORD3], dtype=ctypes.c_int)

    kernel_griddata2D_new(n_active_total, bs1, bs2, bs3, nb1, nb2, nb3, &n_ord_c[0],&input[0,0,0,0], &output[0,0,0,0], ACTIVE_1, ACTIVE2, ACTIVE_3, &AMR_LEVEL1_c[0], &AMR_LEVEL2_c[0], &AMR_LEVEL3_c[0], &AMR_COORD1_c[0], &AMR_COORD2_c[0], &AMR_COORD3_c[0])

def rgdump_write(flag, dir, axisym, np.ndarray[np.int32_t, ndim=1, mode="c"] n_ord,  f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  x1,np.ndarray[np.float32_t, ndim=4, mode="c"]  x2, np.ndarray[np.float32_t, ndim=4, mode="c"]  x3,
np.ndarray[np.float32_t, ndim=4, mode="c"]  r,np.ndarray[np.float32_t, ndim=4, mode="c"]  h, np.ndarray[np.float32_t, ndim=4, mode="c"]  ph,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon,
np.ndarray[np.float32_t, ndim=6, mode="c"]  dxdxp,np.ndarray[np.float32_t, ndim=4, mode="c"]  gdet):
    cdef np.ndarray[int, ndim=1, mode="c"] n_ord_c
    n_ord_c = np.ascontiguousarray(n_ord, dtype=ctypes.c_int)
    kernel_rgdump_write(flag,dir, axisym, &n_ord_c[0], f1, f2, f3, nb, bs1, bs2, bs3, &x1[0,0,0,0],&x2[0,0,0,0],&x3[0,0,0,0],&r[0,0,0,0],&h[0,0,0,0],&ph[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],&dxdxp[0,0,0,0,0,0],&gdet[0,0,0,0])

def rdump_write(flag, RAD_M1, RESISTIVE, TWO_T, P_NUM,dir, dump, n_active_total, f1,f2,f3,nb,bs1,bs2,bs3, np.ndarray[np.float32_t, ndim=4, mode="c"]  rho,np.ndarray[np.float32_t, ndim=4, mode="c"]  ug, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu, np.ndarray[np.float32_t, ndim=5, mode="c"]  B, np.ndarray[np.float32_t, ndim=5, mode="c"]  E,np.ndarray[np.float32_t, ndim=4, mode="c"]  E_rad, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu_rad,np.ndarray[np.float32_t, ndim=4, mode="c"]  TE,np.ndarray[np.float32_t, ndim=4, mode="c"]  TI,np.ndarray[np.float32_t, ndim=4, mode="c"]  photon_number, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov, np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon, axisym):
    kernel_rdump_write(flag, RAD_M1, RESISTIVE, TWO_T, P_NUM, dir, dump, n_active_total, f1, f2, f3, nb, bs1, bs2, bs3, &rho[0,0,0,0], &ug[0,0,0,0], &uu[0,0,0,0,0], &B[0,0,0,0,0],&E[0,0,0,0,0],&E_rad[0,0,0,0],&uu_rad[0,0,0,0,0],&TE[0,0,0,0],&TI[0,0,0,0],&photon_number[0,0,0,0],&gcov[0,0,0,0,0,0],&gcon[0,0,0,0,0,0],axisym)

def misc_calc(bs1, bs2, bs3, nb, axisym, np.ndarray[np.float32_t, ndim=5, mode="c"] uu, np.ndarray[np.float32_t, ndim=5, mode="c"] B,np.ndarray[np.float32_t, ndim=5, mode="c"] E, np.ndarray[np.float32_t, ndim=5, mode="c"] bu,np.ndarray[np.float32_t, ndim=5, mode="c"] eu, np.ndarray[np.float32_t, ndim=6, mode="c"] gcov, np.ndarray[np.float32_t, ndim=4, mode="c"]  bsq, np.ndarray[np.float32_t, ndim=4, mode="c"]  esq, calc_bu, calc_eu, calc_bsq, calc_esq):
    kernel_misc_calc(bs1, bs2, bs3, nb,axisym, &uu[0,0,0,0,0], &B[0,0,0,0,0],&E[0,0,0,0,0], &bu[0,0,0,0,0],&eu[0,0,0,0,0], &gcov[0,0,0,0,0,0], &bsq[0,0,0,0],&esq[0,0,0,0], calc_bu, calc_eu, calc_bsq, calc_esq)

# Calculate tilt and precession angles accurate with need for calc_cart
def calc_precesion_accurate_disk_c(np.ndarray[np.float32_t, ndim=4, mode="c"]  r,np.ndarray[np.float32_t, ndim=4, mode="c"]  h,np.ndarray[np.float32_t, ndim=4, mode="c"]  ph,
np.ndarray[np.float32_t, ndim=4, mode="c"]  rho,np.ndarray[np.float32_t, ndim=4, mode="c"]  ug, np.ndarray[np.float32_t, ndim=5, mode="c"]  uu, np.ndarray[np.float32_t, ndim=5, mode="c"]  B,
np.ndarray[np.float32_t, ndim=6, mode="c"]  dxdxp,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcov,np.ndarray[np.float32_t, ndim=6, mode="c"]  gcon, np.ndarray[np.float32_t, ndim=4, mode="c"]  gdet, avg,
tilt_angle, nb, bs1, bs2, bs3, gam, axisym):

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  J_car = np.zeros((4, nb, bs1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"]  S_r = np.zeros((nb, bs1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  JBH_cross_D = np.zeros((4, nb, bs1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  J_BH = np.zeros((4, nb, bs1), dtype=np.float32, order='C')

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  J_car_avg = np.zeros((4, nb, 1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"]  S_r_avg = np.zeros((nb, 1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  JBH_cross_D_avg = np.zeros((4, nb, 1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  J_BH_avg = np.zeros((4, nb, 1), dtype=np.float32, order='C')

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  Su_disk = np.zeros((4, nb, bs1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"]  L_disk = np.zeros((4, 4, nb, bs1), dtype=np.float32, order='C')

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  Su_corona = np.zeros((4, nb, bs1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"]  L_corona = np.zeros((4, 4, nb, bs1), dtype=np.float32, order='C')

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  Su_disk_avg = np.zeros((4, nb, 1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"]  L_disk_avg = np.zeros((4, 4, nb, 1), dtype=np.float32, order='C')

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"]  Su_corona_avg = np.zeros((4, nb, 1), dtype=np.float32, order='C')
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"]  L_corona_avg = np.zeros((4, 4, nb, 1), dtype=np.float32, order='C')

    tilt = tilt_angle / 180. * 3.141592

    kernel_calc_prec_disk(bs1, bs2, bs3, nb,axisym, avg, &r[0,0,0,0], &h[0,0,0,0], &ph[0,0,0,0], &rho[0,0,0,0],&ug[0,0,0,0], &uu[0,0,0,0,0],&B[0,0,0,0,0],gam, &gcov[0,0,0,0,0,0], &gcon[0,0,0,0,0,0], &gdet[0,0,0,0], &dxdxp[0,0,0,0,0,0], &Su_disk[0,0,0], &L_disk[0,0,0,0],&Su_corona[0,0,0], &L_corona[0,0,0,0] ,&Su_disk_avg[0,0,0], &L_disk_avg[0,0,0,0],&Su_corona_avg[0,0,0], &L_corona_avg[0,0,0,0])

    #Calculate for disk
    S_r[0] = Su_disk[0, 0] * Su_disk[0, 0] + Su_disk[1, 0] * Su_disk[1, 0] + Su_disk[2, 0] * Su_disk[2, 0] + Su_disk[3, 0] * Su_disk[3, 0]
    J_car[3] = (L_disk[1, 2] * Su_disk[0] - L_disk[2, 1] * Su_disk[0] + L_disk[2, 0] * Su_disk[1] - L_disk[0, 2] * Su_disk[1] - L_disk[1, 0] * Su_disk[2] + L_disk[0, 1] * Su_disk[2]) / (2 * np.sqrt(np.abs(-S_r)))
    J_car[2] = -(L_disk[1, 3] * Su_disk[0] - L_disk[3, 1] * Su_disk[0] - L_disk[1, 0] * Su_disk[3] + L_disk[3, 0] * Su_disk[1] - L_disk[0, 3] * Su_disk[1] + L_disk[0, 1] * Su_disk[3]) / (2 * np.sqrt(np.abs(-S_r)))
    J_car[1] = (L_disk[2, 3] * Su_disk[0] - L_disk[3, 2] * Su_disk[0] - L_disk[2, 0] * Su_disk[3] + L_disk[0, 2] * Su_disk[3] + L_disk[3, 0] * Su_disk[2] - L_disk[0, 3] * Su_disk[ 2]) / (2 * np.sqrt(np.abs(-S_r)))
    J_length = np.sqrt(J_car[1] * J_car[1] + J_car[2] * J_car[2] + J_car[3] * J_car[3])

    J_BH[1] = -np.sin(tilt) * J_car[1] / J_car[1]
    J_BH[2] = 0 * J_car[1] / J_car[1]
    J_BH[3] = np.cos(tilt) * J_car[1] / J_car[1]
    J_BH_length = np.sqrt(J_BH[1] * J_BH[1] + J_BH[2] * J_BH[2] + J_BH[3] * J_BH[3])

    JBH_cross_D[1] = J_BH[2] * J_car[3] - J_BH[3] * J_car[2]
    JBH_cross_D[2] = J_BH[3] * J_car[1] - J_BH[1] * J_car[3]
    JBH_cross_D[3] = J_BH[1] * J_car[2] - J_BH[2] * J_car[1]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    angle_tilt_disk = np.arccos(np.abs(J_car[1] * J_BH[1] + J_car[2] * J_BH[2] + J_car[3] * J_BH[3]) / (J_BH_length * J_length)) * 180.0 / np.pi
    angle_prec_disk = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180.0 / np.pi

    #Calculate for disk average
    S_r_avg[0] = -Su_disk_avg[0, 0] * Su_disk_avg[0, 0] + Su_disk_avg[1, 0] * Su_disk_avg[1, 0] + Su_disk_avg[2, 0] * Su_disk_avg[2, 0] + Su_disk_avg[3, 0] * Su_disk_avg[3, 0]
    #J_car_avg[3] = (L_disk_avg[1, 2] * Su_disk_avg[0] - L_disk_avg[2, 1] * Su_disk_avg[0] + L_disk_avg[2, 0] * Su_disk_avg[1] - L_disk_avg[0, 2] * Su_disk_avg[1] - L_disk_avg[1, 0] * Su_disk_avg[2] + L_disk_avg[0, 1] * Su_disk_avg[2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    #J_car_avg[2] = -(L_disk_avg[1, 3] * Su_disk_avg[0] - L_disk_avg[3, 1] * Su_disk_avg[0] - L_disk_avg[1, 0] * Su_disk_avg[3] + L_disk_avg[3, 0] * Su_disk_avg[1] - L_disk_avg[0, 3] * Su_disk_avg[1] + L_disk_avg[0, 1] * Su_disk_avg[3]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    #J_car_avg[1] = (L_disk_avg[2, 3] * Su_disk_avg[0] - L_disk_avg[3, 2] * Su_disk_avg[0] - L_disk_avg[2, 0] * Su_disk_avg[3] + L_disk_avg[0, 2] * Su_disk_avg[3] + L_disk_avg[3, 0] * Su_disk_avg[2] - L_disk_avg[0, 3] * Su_disk_avg[ 2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    J_car_avg[3] = (L_disk_avg[1, 2])
    J_car_avg[2] = -(L_disk_avg[1, 3])
    J_car_avg[1] = (L_disk_avg[2, 3])
    J_length_avg = np.sqrt(J_car_avg[1] * J_car_avg[1] + J_car_avg[2] * J_car_avg[2] + J_car_avg[3] * J_car_avg[3])

    J_BH_avg[1] = -np.sin(tilt) * J_car_avg[1] / J_car_avg[1]
    J_BH_avg[2] = 0 * J_car_avg[1] / J_car_avg[1]
    J_BH_avg[3] = np.cos(tilt) * J_car_avg[1] / J_car_avg[1]
    J_BH_length_avg = np.sqrt(J_BH_avg[1] * J_BH_avg[1] + J_BH_avg[2] * J_BH_avg[2] + J_BH_avg[3] * J_BH_avg[3])

    JBH_cross_D_avg[1] = J_BH_avg[2] * J_car_avg[3] - J_BH_avg[3] * J_car_avg[2]
    JBH_cross_D_avg[2] = J_BH_avg[3] * J_car_avg[1] - J_BH_avg[1] * J_car_avg[3]
    JBH_cross_D_avg[3] = J_BH_avg[1] * J_car_avg[2] - J_BH_avg[2] * J_car_avg[1]
    JBH_cross_D_length_avg = np.sqrt(JBH_cross_D_avg[1] * JBH_cross_D_avg[1] + JBH_cross_D_avg[2] * JBH_cross_D_avg[2] + JBH_cross_D_avg[3] * JBH_cross_D_avg[3])

    angle_tilt_disk_avg = np.arccos(np.abs(J_car_avg[1] * J_BH_avg[1] + J_car_avg[2] * J_BH_avg[2] + J_car_avg[3] * J_BH_avg[3]) / (J_BH_length_avg * J_length_avg)) * 180.0 / np.pi
    angle_prec_disk_avg = -np.arctan2(JBH_cross_D_avg[1], JBH_cross_D_avg[2]) * 180.0 / np.pi

    #Calculate for corona
    S_r[0] = Su_corona[0, 0] * Su_corona[0, 0] + Su_corona[1, 0] * Su_corona[1, 0] + Su_corona[2, 0] * Su_corona[2, 0] + Su_corona[3, 0] * Su_corona[3, 0]
    J_car[3] = (L_corona[1, 2] * Su_corona[0] - L_corona[2, 1] * Su_corona[0] + L_corona[2, 0] * Su_corona[1] - L_corona[0, 2] * Su_corona[1] - L_corona[1, 0] * Su_corona[2] + L_corona[0, 1] * Su_corona[2]) / (2 * np.sqrt(np.abs(-S_r)))
    J_car[2] = -(L_corona[1, 3] * Su_corona[0] - L_corona[3, 1] * Su_corona[0] - L_corona[1, 0] * Su_corona[3] + L_corona[3, 0] * Su_corona[1] - L_corona[0, 3] * Su_corona[1] + L_corona[0, 1] * Su_corona[3]) / (2 * np.sqrt(np.abs(-S_r)))
    J_car[1] = (L_corona[2, 3] * Su_corona[0] - L_corona[3, 2] * Su_corona[0] - L_corona[2, 0] * Su_corona[3] + L_corona[0, 2] * Su_corona[3] + L_corona[3, 0] * Su_corona[2] - L_corona[0, 3] * Su_corona[ 2]) / (2 * np.sqrt(np.abs(-S_r)))
    J_length = np.sqrt(J_car[1] * J_car[1] + J_car[2] * J_car[2] + J_car[3] * J_car[3])

    J_BH[1] = -np.sin(tilt) * J_car[1] / J_car[1]
    J_BH[2] = 0 * J_car[1] / J_car[1]
    J_BH[3] = np.cos(tilt) * J_car[1] / J_car[1]
    J_BH_length = np.sqrt(J_BH[1] * J_BH[1] + J_BH[2] * J_BH[2] + J_BH[3] * J_BH[3])

    JBH_cross_D[1] = J_BH[2] * J_car[3] - J_BH[3] * J_car[2]
    JBH_cross_D[2] = J_BH[3] * J_car[1] - J_BH[1] * J_car[3]
    JBH_cross_D[3] = J_BH[1] * J_car[2] - J_BH[2] * J_car[1]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    angle_tilt_corona = np.arccos(np.abs(J_car[1] * J_BH[1] + J_car[2] * J_BH[2] + J_car[3] * J_BH[3]) / (J_BH_length * J_length)) * 180.0 / np.pi
    angle_prec_corona = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180.0 / np.pi

    #Calculate for corona average
    S_r_avg[0] = Su_corona_avg[0, 0] * Su_corona_avg[0, 0] + Su_corona_avg[1, 0] * Su_corona_avg[1, 0] + Su_corona_avg[2, 0] * Su_corona_avg[2, 0] + Su_corona_avg[3, 0] * Su_corona_avg[3, 0]
    J_car_avg[3] = (L_corona_avg[1, 2] * Su_corona_avg[0] - L_corona_avg[2, 1] * Su_corona_avg[0] + L_corona_avg[2, 0] * Su_corona_avg[1] - L_corona_avg[0, 2] * Su_corona_avg[1] - L_corona_avg[1, 0] * Su_corona_avg[2] + L_corona_avg[0, 1] * Su_corona_avg[2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    J_car_avg[2] = -(L_corona_avg[1, 3] * Su_corona_avg[0] - L_corona_avg[3, 1] * Su_corona_avg[0] - L_corona_avg[1, 0] * Su_corona_avg[3] + L_corona_avg[3, 0] * Su_corona_avg[1] - L_corona_avg[0, 3] * Su_corona_avg[1] + L_corona_avg[0, 1] * Su_corona_avg[3]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    J_car_avg[1] = (L_corona_avg[2, 3] * Su_corona_avg[0] - L_corona_avg[3, 2] * Su_corona_avg[0] - L_corona_avg[2, 0] * Su_corona_avg[3] + L_corona_avg[0, 2] * Su_corona_avg[3] + L_corona_avg[3, 0] * Su_corona_avg[2] - L_corona_avg[0, 3] * Su_corona_avg[ 2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
    J_length_avg = np.sqrt(J_car_avg[1] * J_car_avg[1] + J_car_avg[2] * J_car_avg[2] + J_car_avg[3] * J_car_avg[3])

    J_BH_avg[1] = -np.sin(tilt) * J_car_avg[1] / J_car_avg[1]
    J_BH_avg[2] = 0 * J_car_avg[1] / J_car_avg[1]
    J_BH_avg[3] = np.cos(tilt) * J_car_avg[1] / J_car_avg[1]
    J_BH_length_avg = np.sqrt(J_BH_avg[1] * J_BH_avg[1] + J_BH_avg[2] * J_BH_avg[2] + J_BH_avg[3] * J_BH_avg[3])

    JBH_cross_D_avg[1] = J_BH_avg[2] * J_car_avg[3] - J_BH_avg[3] * J_car_avg[2]
    JBH_cross_D_avg[2] = J_BH_avg[3] * J_car_avg[1] - J_BH_avg[1] * J_car_avg[3]
    JBH_cross_D_avg[3] = J_BH_avg[1] * J_car_avg[2] - J_BH_avg[2] * J_car_avg[1]
    JBH_cross_D_length_avg = np.sqrt(JBH_cross_D_avg[1] * JBH_cross_D_avg[1] + JBH_cross_D_avg[2] * JBH_cross_D_avg[2] + JBH_cross_D_avg[3] * JBH_cross_D_avg[3])

    angle_tilt_corona_avg = np.arccos(np.abs(J_car_avg[1] * J_BH_avg[1] + J_car_avg[2] * J_BH_avg[2] + J_car_avg[3] * J_BH_avg[3]) / (J_BH_length_avg * J_length_avg)) * 180.0 / np.pi
    angle_prec_corona_avg = -np.arctan2(JBH_cross_D_avg[1], JBH_cross_D_avg[2]) * 180.0 / np.pi

    return angle_tilt_disk,angle_prec_disk,angle_tilt_corona,angle_prec_corona,angle_tilt_disk_avg,angle_prec_disk_avg,angle_tilt_corona_avg,angle_prec_corona_avg
