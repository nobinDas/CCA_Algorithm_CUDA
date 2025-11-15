#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// Device function declarations for utility and union-find operations
__device__ int getIndex(int x, int y, int width);
__device__ int findRoot(int *labels, int idx);
__device__ void merge(int *labels, int a, int b);

// Phase 1: Local labeling within blocks
__global__ void optimized_ccl_kernel(unsigned char *img, int *labels, int width, int height);

// Phase 2: Boundary merge across blocks 
__global__ void boundary_merge_kernel(unsigned char *img, int *labels, int width, int height);

// Phase 3: Final label compression
__global__ void compress_labels_kernel(int *labels, int width, int height);

#endif // KERNEL_CUH