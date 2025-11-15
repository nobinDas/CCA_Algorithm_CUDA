// Complete CUDA Connected Component Labeling Implementation
// Three-phase algorithm: Local Labeling + Boundary Merge + Label Compression

#include <stdio.h>
#include "kernel.cuh"

__device__ int getIndex(int x, int y, int width) {
    return y * width + x;
}

// Union-find operations for boundary merge phase
__device__ int findRoot(int *labels, int idx) {
    int root = idx;
    while (labels[root] != root && labels[root] != -1) {
        root = labels[root];
    }
    return root;
}

__device__ void merge(int *labels, int a, int b) {
    int rootA = findRoot(labels, a);
    int rootB = findRoot(labels, b);
    if (rootA != rootB && rootA != -1 && rootB != -1) {
        // Union by making smaller root point to larger root
        if (rootA < rootB) {
            atomicMin(&labels[rootB], rootA);
        } else {
            atomicMin(&labels[rootA], rootB);
        }
    }
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

// Boundary Merge Implementation - merges components across block boundaries
__global__ void boundary_merge_kernel(unsigned char *img, int *labels, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int idx = row * width + col;
    
    // Only process foreground pixels at block boundaries
    if (img[idx] == 1) {
        int blockDimX = 16; // Block size
        int blockDimY = 16;
        
        // Check if pixel is at vertical block boundary (merge with left neighbor)
        if (col % blockDimX == 0 && col > 0) {
            int leftIdx = idx - 1;
            if (img[leftIdx] == 1 && labels[leftIdx] != labels[idx]) {
                merge(labels, leftIdx, idx);
            }
        }
        
        // Check if pixel is at horizontal block boundary (merge with upper neighbor) 
        if (row % blockDimY == 0 && row > 0) {
            int upIdx = idx - width;
            if (img[upIdx] == 1 && labels[upIdx] != labels[idx]) {
                merge(labels, upIdx, idx);
            }
        }
    }
}

// Label Compression Implementation - resolves all label equivalences
__global__ void compress_labels_kernel(int *labels, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx < totalPixels) {
        if (labels[idx] != -1) {  // Only process foreground pixels
            labels[idx] = findRoot(labels, idx);
        }
    }
}