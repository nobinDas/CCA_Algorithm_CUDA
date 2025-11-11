// Modern Block-based Union Find Algorithm for Connected Components
// Based on Allegretti et al. 2019 - YACCLAB BUF algorithm
// Simplified implementation for educational purposes

#include "kernel.cuh"

#define BLOCK_SIZE_2D 16

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
    
    if (img[idx] == 0) {
        labels[idx] = -1; // Background
        return;
    }
    
    // Initialize with own index
    labels[idx] = idx;
    
    __syncthreads();
    
    // Check neighbors and union if connected
    // Left neighbor
    if (col > 0 && img[idx - 1] == 1) {
        union_labels(labels, idx, idx - 1);
    }
    
    // Top neighbor  
    if (row > 0 && img[idx - width] == 1) {
        union_labels(labels, idx, idx - width);
    }
    
    // Diagonal neighbors for 8-connectivity
    if (row > 0 && col > 0 && img[idx - width - 1] == 1) {
        union_labels(labels, idx, idx - width - 1);
    }
    
    if (row > 0 && col < width - 1 && img[idx - width + 1] == 1) {
        union_labels(labels, idx, idx - width + 1);
    }
    
    __syncthreads();
    
    // Final compression pass
    if (img[idx] == 1) {
        labels[idx] = find_root(labels, idx);
    }
}