#include <stdio.h>
#include "kernel.cuh"

__device__ int getIndex(int x, int y, int width) {
    return y * width + x;
}

__global__ void initializeLabels(unsigned char *img, int *labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = getIndex(x, y, width);
    if (img[idx] == 0) {
        labels[idx] = -1; // Background
    } else {
        labels[idx] = idx; // Each foreground pixel gets its own initial label
    }
}

__global__ void labelPropagation(unsigned char *img, int *labels, int *changed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = getIndex(x, y, width);
    if (img[idx] == 0) return; // Skip background pixels

    int currentLabel = labels[idx];
    int minLabel = currentLabel;

    // Check 8-connected neighbors
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIdx = getIndex(nx, ny, width);
                if (img[nIdx] == 1 && labels[nIdx] >= 0) {
                    minLabel = min(minLabel, labels[nIdx]);
                }
            }
        }
    }

    if (minLabel < currentLabel) {
        labels[idx] = minLabel;
        *changed = 1;
    }
}

__global__ void connectedComponentLabeling(unsigned char *img, int *labels, int width, int height) {
    // This is a simplified single-kernel approach
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = getIndex(x, y, width);
    if (img[idx] == 0) {
        labels[idx] = -1;
        return;
    }

    // Initialize with own index
    labels[idx] = idx;
    
    __syncthreads();

    // Fixed number of iterations for label propagation
    for (int iter = 0; iter < 20; iter++) {
        int currentLabel = labels[idx];
        int minLabel = currentLabel;

        // Check 4-connected neighbors
        if (x > 0 && img[getIndex(x - 1, y, width)] == 1) {
            minLabel = min(minLabel, labels[getIndex(x - 1, y, width)]);
        }
        if (y > 0 && img[getIndex(x, y - 1, width)] == 1) {
            minLabel = min(minLabel, labels[getIndex(x, y - 1, width)]);
        }
        if (x < width - 1 && img[getIndex(x + 1, y, width)] == 1) {
            minLabel = min(minLabel, labels[getIndex(x + 1, y, width)]);
        }
        if (y < height - 1 && img[getIndex(x, y + 1, width)] == 1) {
            minLabel = min(minLabel, labels[getIndex(x, y + 1, width)]);
        }

        labels[idx] = minLabel;
        __syncthreads();
    }
}

// Modern Block-based Union Find Algorithm
__device__ unsigned find_root(int* labels, unsigned n) {
    while (labels[n] != n) {
        // Path compression for optimization
        unsigned next = labels[n];
        labels[n] = labels[labels[n]]; 
        n = next;
    }
    return n;
}

__device__ void union_labels(int* labels, unsigned a, unsigned b) {
    bool done;
    do {
        a = find_root(labels, a);
        b = find_root(labels, b);
        
        if (a < b) {
            int old = atomicMin(labels + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a) {
            int old = atomicMin(labels + a, b);
            done = (old == a); 
            a = old;
        }
        else {
            done = true;
        }
    } while (!done);
}

__global__ void modern_ccl_kernel(unsigned char *img, int *labels, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int idx = row * width + col;
    
    // Phase 1: Quick initialization - no unnecessary branching
    labels[idx] = (img[idx] == 0) ? -1 : idx;
    
    __syncthreads();
    
    // Phase 2: Efficient label propagation - reduced iterations
    // Using only backward neighbors for better convergence
    for (int iter = 0; iter < 4; iter++) {
        if (img[idx] == 1) {
            int current = labels[idx];
            int minimum = current;
            
            // Check only left and top neighbors (efficient scanning order)
            if (col > 0 && img[idx - 1] == 1) {
                minimum = min(minimum, labels[idx - 1]);
            }
            if (row > 0 && img[idx - width] == 1) {
                minimum = min(minimum, labels[idx - width]);
            }
            
            labels[idx] = minimum;
        }
        __syncthreads();
    }
    
    // Phase 3: Forward pass for missed connections
    for (int iter = 0; iter < 2; iter++) {
        if (img[idx] == 1) {
            int current = labels[idx];
            int minimum = current;
            
            // Check forward neighbors
            if (col < width - 1 && img[idx + 1] == 1) {
                minimum = min(minimum, labels[idx + 1]);
            }
            if (row < height - 1 && img[idx + width] == 1) {
                minimum = min(minimum, labels[idx + width]);
            }
            
            labels[idx] = minimum;
        }
        __syncthreads();
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