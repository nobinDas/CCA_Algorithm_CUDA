#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// Device function declaration for utility
__device__ int getIndex(int x, int y, int width);

// Main GPU kernel function used in the project
__global__ void optimized_ccl_kernel(unsigned char *img, int *labels, int width, int height);

#endif // KERNEL_CUH