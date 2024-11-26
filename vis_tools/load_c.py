from ctypes import *
from numpy.ctypeslib import ndpointer

## Load C libraries from shared object files
'''
To do:
    1. Change absolute paths
    2. Check that paths exist
'''

## Load libfunctions
cdll.LoadLibrary("./pp_c.cpython-39-x86_64-linux-gnu.so")
libfunctions = CDLL("./pp_c.cpython-39-x86_64-linux-gnu.so") 
#cdll.LoadLibrary("/pscratch/sd/n/nkaaz/analysis/harm2d/libs/libfunctions.so")
#libfunctions = CDLL("/pscratch/sd/n/nkaaz/analysis/harm2d/libs/libfunctions.so")  

'''
    Convert each desired C function from loaded libraries into Python functions.
    Steps:
        1. python_func = library.c_function
        2. python_func.restype = None
        3. python_func.argtypes = [,,,]
        
    Step 3 requires the most work, need to match each argument with argument in 'c_function'
    C integer --> "c_int"
    C float   --> "c_float"
    C string  --> "c_char_p" (need to pass string as:  c_char_p(string.encode('utf-8'))
    C pointer --> "ndpointer(data_Type, flags="C_CONTIGUOUS")"
        
        
'''


'''
functions
'''


# C:  libfunctions.kernel_rgdump_new
# py: kernel_read_gdumps
kernel_read_gdumps = libfunctions.kernel_rgdump_new
kernel_read_gdumps.restype = None
kernel_read_gdumps.argtypes = [c_int, c_char_p, c_int,
                ndpointer(c_int, flags="C_CONTIGUOUS"),
                c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS"),                
                ndpointer(c_float, flags="C_CONTIGUOUS"),
                ndpointer(c_float, flags="C_CONTIGUOUS")]

kernel_read_dump = libfunctions.kernel_rdump_new
kernel_read_dump.restype = None
kernel_read_dump.argtypes = [c_int, c_int, c_int, c_int, c_int, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"),
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"),
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"),
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), 
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), 
                            ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]

# C:  functions.kernel_griddata3D_new
# py: kernel_griddata3D
kernel_griddata3D = libfunctions.kernel_griddata3D_new
kernel_griddata3D.restype  = None
kernel_griddata3D.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"),
                              ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), 
                              ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS")]

# C:  functions.kernel_griddata2D_new
# py: kernel_griddata2D
kernel_griddata2D = libfunctions.kernel_griddata2D_new
kernel_griddata2D.restype  = None
kernel_griddata2D.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"),
                              ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), 
                              ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS")]

# C:  functions.kernel_rgdump_griddata
# py: kernel_read_gdumps_griddata
kernel_read_gdumps_griddata = libfunctions.kernel_rgdump_griddata
kernel_read_gdumps_griddata.restype  = None
kernel_read_gdumps_griddata.argtypes = [c_int, c_int, c_char_p, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

# C:  functions.kernel_rdump_griddata
# py: kernel_read_dump_griddata
kernel_read_dump_griddata = libfunctions.kernel_rdump_griddata
kernel_read_dump_griddata.restype  = None
kernel_read_dump_griddata.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), c_int, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), ndpointer(c_int, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float,  ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_float, c_float, c_float, c_float, c_float, c_float,  ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int]

# C: functions.kernel_calc_prec_disk
# py: kernel_calc_warp
kernel_calc_warp = libfunctions.kernel_calc_prec_disk
kernel_calc_warp.restype  = None
kernel_calc_warp.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_float, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS")]

# C: functions.kernel_invert_4x4
# py: kernel_invert_4x4
kernel_invert_4x4 = libfunctions.kernel_invert_4x4
kernel_invert_4x4.restype  = None
kernel_invert_4x4.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int] 

