#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef __unix__
#include <sys/time.h>
#endif
#ifndef __APPLE__
#include <omp.h>
#endif
#include "config.h"
#if(SCUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif(SHIP)
#include "hip/hip_runtime.h"
#endif