// Clean CUDA Connected Component Labeling Implementation
// Contains only the optimized algorithm currently used in main.cu

#include <stdio.h>
#include "kernel.cuh"

__device__ int getIndex(int x, int y, int width) {
    return y * width + x;
}

// High-performance block-based CCL kernel
__global__ void optimized_ccl_kernel(unsigned char *img, int *labels, int width, int height) {
    // Use shared memory for block-local processing
    __shared__ int s_labels[16*16];
    __shared__ unsigned char s_img[16*16];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;
    
    // Load data into shared memory
    if (row < height && col < width) {
        s_img[tid] = img[idx];
        s_labels[tid] = (img[idx] == 0) ? -1 : idx;
    } else {
        s_img[tid] = 0;
        s_labels[tid] = -1;
    }
    
    __syncthreads();
    
    // Process within block using shared memory
    if (s_img[tid] == 1) {
        for (int iter = 0; iter < 6; iter++) {
            int current = s_labels[tid];
            int minimum = current;
            
            int local_row = threadIdx.y;
            int local_col = threadIdx.x;
            
            // Check neighbors within block
            if (local_col > 0 && s_img[tid - 1] == 1) {
                minimum = min(minimum, s_labels[tid - 1]);
            }
            if (local_row > 0 && s_img[tid - blockDim.x] == 1) {
                minimum = min(minimum, s_labels[tid - blockDim.x]);
            }
            if (local_col < blockDim.x - 1 && s_img[tid + 1] == 1) {
                minimum = min(minimum, s_labels[tid + 1]);
            }
            if (local_row < blockDim.y - 1 && s_img[tid + blockDim.x] == 1) {
                minimum = min(minimum, s_labels[tid + blockDim.x]);
            }
            
            s_labels[tid] = minimum;
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (row < height && col < width) {
        labels[idx] = s_labels[tid];
    }
}