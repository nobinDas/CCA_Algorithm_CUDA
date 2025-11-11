#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// Device function declarations
__device__ int getIndex(int x, int y, int width);
__device__ unsigned find_root(int* labels, unsigned n);
__device__ void union_labels(int* labels, unsigned a, unsigned b);

// Kernel function declarations
__global__ void initializeLabels(unsigned char *img, int *labels, int width, int height);
__global__ void labelPropagation(unsigned char *img, int *labels, int *changed, int width, int height);
__global__ void connectedComponentLabeling(unsigned char *img, int *labels, int width, int height);
__global__ void modern_ccl_kernel(unsigned char *img, int *labels, int width, int height);
__global__ void optimized_ccl_kernel(unsigned char *img, int *labels, int width, int height);

// CPU function for comparison
void cpuConnectedComponents(unsigned char *img, int *labels, int width, int height);

#endif // KERNEL_CUH