#### set USEICC to 0 if you want gcc compiler options, else set to 1 to use icc
########  gcc generally used for debugging with -g option so we can use gdb 
USEICC = 0

ifeq ($(USEICC),0)
CC       = CC 
CCFLAGS  = -I${ROCM_PATH}/include -std=c++11 -D CRAY_CPU_TARGET=x86-64 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -fgpu-rdc -x hip -O1 
endif

EXTRALIBS = -lm -fgpu-rdc --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64 -lstdc++

CC_COMPILE  = $(CC) $(CCFLAGS) -c 
CUDA_COMPILE  = hipcc -fgpu-rdc --amdgpu-target=gfx90a -I${MPICH_DIR}/include -I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__ -fopenmp -O1 -c
CC_LOAD     = $(CC) $(CCFLAGS)
CUDA_LOAD  = hipcc -v -fgpu-rdc --amdgpu-target=gfx90a -L${MPICH_DIR}/lib -lmpi -L${MPICH_DIR}/lib -lmpi ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -fopenmp

GPU_FILES = GPU_boundcomP.cu GPU_boundcomF.cu GPU_boundcomE.cu GPU_main.cu GPU_program1.cu GPU_program2.cu

.c.o:
	$(CUDA_COMPILE) $*.c

EXE = harm
all: $(EXE)
	
OBJS = \
AMR.o boundcomB.o boundcomE.o boundcomF.o boundcomP.o \
bounds.o coord.o const_trans.o const_trans_res.o diag.o dump.o eos_helm.o fixup.o\
GPU_boundcomE.o GPU_boundcomP.o GPU_boundcomF.o GPU_program1.o GPU_program2.o GPU_main.o\
hllc.o LAS.o init_collapsar.o init_mag.o init_misc.o init_nsm.o init_tde.o \
init_tests.o init_thindisk.o init_torus.o init_torus_grb.o init_torus_spherical.o \
interp.o lu.o main.o memory.o metric.o phys_2T.o phys_mhd.o phys_neutrinos.o \
phys_nu.o phys_nuclear.o phys_radiation.o phys_res.o radiation.o ranc.o restart.o step_ch.o step_ch_res.o \
u2p_util.o utoprim_1dfix1.o utoprim_1dvsq2fix1.o utoprim_2d.o utoprim_3d_res.o utoprim_nm.o wrapper.o

INCS = \
decs.h decs_MPI.h decsCUDA.h defs.h include.h u2p_defs.h u2p_util.h config.h

$(OBJS) : $(INCS) makefile

$(EXE): $(OBJS) $(INCS) makefile
	$(CUDA_COMPILE) $(GPU_FILES)
	$(CUDA_LOAD) $(OBJS) -o $(EXE)

clean:
	/bin/rm -f *.o *.il
	/bin/rm -f $(EXE) image_interp
