#include "include.h"
#include "decs.h"

gpuError_t gpuMalloc(void** ptr, size_t size) {
#if(SHIP)
	return hipMalloc(ptr, size);
#elif(SCUDA)
	return cudaMalloc(ptr, size);
#endif
}

gpuError_t gpuMallocHost(void** ptr, size_t size) {
#if(SHIP)
	return hipHostMalloc(ptr, size, hipHostMallocMapped);
#elif(SCUDA)
	return cudaMallocHost(ptr, size);
#endif
}

gpuError_t gpuFreeHost(void* ptr) {
#if(SHIP)
	return hipHostFree(ptr);
#elif(SCUDA)
	return cudaFreeHost(ptr);
#endif
}

gpuError_t gpuFree(void* ptr) {
#if(SHIP)
	return hipFree(ptr);
#elif(SCUDA)
	return cudaFree(ptr);
#endif
}

gpuError_t gpuGetLastError() {
#if(SHIP)
	return hipGetLastError();
#elif(SCUDA)
	return cudaGetLastError();
#endif
}

gpuError_t gpuSetDevice(int deviceId) {
#if(SHIP)
	return hipSetDevice(deviceId);
#elif(SCUDA)
	return cudaSetDevice(deviceId);
#endif
}

gpuError_t gpuDeviceSetCacheConfig(int cache) {
#if(SHIP)
	return hipDeviceSetCacheConfig((hipFuncCache_t)cache);
#elif(SCUDA)
	return cudaDeviceSetCacheConfig(cache);
#endif
}

gpuError_t gpuDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
#if(SHIP)
	return hipDeviceEnablePeerAccess(peerDevice, flags);
#elif(SCUDA)
	return cudaDeviceEnablePeerAccess(peerDevice, flags);
#endif
}

gpuError_t gpuDeviceSynchronize() {
#if(SHIP)
	return hipDeviceSynchronize();
#elif(SCUDA)
	return cudaDeviceSynchronize();
#endif
}

gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, int kind, gpuStream_t stream) {
#if(SHIP)
	return hipMemcpyAsync(dst, src, count, (hipMemcpyKind)kind, stream);
#elif(SCUDA)
	return cudaMemcpyAsync(dst, src, count, kind, stream);
#endif
}

gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, int kind) {
#if(SHIP)
	return hipMemcpy(dst, src, count, (hipMemcpyKind)kind);
#elif(SCUDA)
	return cudaMemcpy(dst, src, count, kind);
#endif
}

gpuError_t gpuStreamDestroy(gpuStream_t stream) {
#if(SHIP)
	return hipStreamDestroy(stream);
#elif(SCUDA)
	return cudaStreamDestroy(stream);
#endif
}

gpuError_t gpuEventDestroy(gpuEvent_t event) {
#if(SHIP)
	return hipEventDestroy(event);
#elif(SCUDA)
	return cudaEventDestroy(event);
#endif
}

gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
#if(SHIP)
	return hipStreamSynchronize(stream);
#elif(SCUDA)
	return cudaStreamSynchronize(stream);
#endif
}

gpuError_t gpuStreamCreate(gpuStream_t* stream) {
#if(SHIP)
	return hipStreamCreate(stream);
#elif(SCUDA)
	return cudaStreamCreate(stream);
#endif
}

gpuError_t gpuEventCreate(gpuEvent_t* event) {
#if(SHIP)
	return hipEventCreate(event);
#elif(SCUDA)
	return cudaEventCreate(event);
#endif
}

gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
#if(SHIP)
	return hipEventRecord(event, stream);
#elif(SCUDA)
	return cudaEventRecord(event, stream);
#endif
}

gpuError_t gpuStreamWaitEvent(gpuStream_t stream, gpuEvent_t event, int zero) {
#if(SHIP)
	return hipStreamWaitEvent(stream, event, zero);
#elif(SCUDA)
	return cudaStreamWaitEvent(stream, event, zero);
#endif
}

gpuError_t gpuGetDeviceCount(int* count) {
#if(SHIP)
	return hipGetDeviceCount(count);
#elif(SCUDA)
	return cudaGetDeviceCount(count);
#endif
}

gpuError_t gpuDeviceSetSharedMemConfig(int kind) {
#if(SHIP)
	return hipDeviceSetSharedMemConfig((hipSharedMemConfig)kind);
#elif(SCUDA)
	return cudaDeviceSetSharedMemConfig(kind);
#endif
}

gpuError_t gpuMemGetInfo(size_t *free, size_t *total) {
#if(SHIP)
	return hipMemGetInfo(free, total);
#elif(SCUDA)
	return cudaMemGetInfo(free, total);
#endif
}