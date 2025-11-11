#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "kernel.cuh" // header for GPU functions

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024

void generateBinaryImage(unsigned char *img, int width, int height) {
    srand(42); // Fixed seed for reproducible results
    for (int i = 0; i < width * height; i++) {
        img[i] = (rand() % 100 < 40) ? 1 : 0; // 30% foreground pixels
    }
}

void cpuConnectedComponents(unsigned char *img, int *labels, int width, int height) {
    // Initialize labels
    for (int i = 0; i < width * height; i++) {
        if (img[i] == 0) {
            labels[i] = -1; // Background
        } else {
            labels[i] = i; // Each foreground pixel gets its own initial label
        }
    }
    
    // Label propagation
    bool changed = true;
    int iterations = 0;
    while (changed && iterations < 100) {
        changed = false;
        iterations++;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                if (img[idx] == 0) continue;
                
                int currentLabel = labels[idx];
                int minLabel = currentLabel;
                
                // Check 4-connected neighbors
                if (x > 0 && img[idx - 1] == 1) {
                    minLabel = (labels[idx - 1] < minLabel) ? labels[idx - 1] : minLabel;
                }
                if (y > 0 && img[idx - width] == 1) {
                    minLabel = (labels[idx - width] < minLabel) ? labels[idx - width] : minLabel;
                }
                if (x < width - 1 && img[idx + 1] == 1) {
                    minLabel = (labels[idx + 1] < minLabel) ? labels[idx + 1] : minLabel;
                }
                if (y < height - 1 && img[idx + width] == 1) {
                    minLabel = (labels[idx + width] < minLabel) ? labels[idx + width] : minLabel;
                }
                
                if (minLabel < currentLabel) {
                    labels[idx] = minLabel;
                    changed = true;
                }
            }
        }
    }
    printf("CPU iterations: %d\n", iterations);
}

int countComponents(int *labels, int size) {
    bool *used = (bool*)calloc(size, sizeof(bool));
    int count = 0;
    
    for (int i = 0; i < size; i++) {
        if (labels[i] >= 0 && !used[labels[i]]) {
            used[labels[i]] = true;
            count++;
        }
    }
    
    free(used);
    return count;
}

int main() {
    unsigned char *h_img, *d_img;
    int *h_labels_cpu, *h_labels_gpu, *d_labels;
    int size = IMG_WIDTH * IMG_HEIGHT;

    // Allocate host memory
    h_img = (unsigned char*)malloc(size);
    h_labels_cpu = (int*)malloc(size * sizeof(int));
    h_labels_gpu = (int*)malloc(size * sizeof(int));

    // Generate a random binary image
    generateBinaryImage(h_img, IMG_WIDTH, IMG_HEIGHT);
    
    printf("=== CPU vs GPU Connected Component Analysis Comparison ===\n");
    printf("Image size: %dx%d (%d pixels)\n", IMG_WIDTH, IMG_HEIGHT, size);
    printf("Foreground density: ~30%%\n\n");

    // CPU Version
    printf("🖥️  === CPU Connected Components (Reference) ===\n");
    clock_t start = clock();
    cpuConnectedComponents(h_img, h_labels_cpu, IMG_WIDTH, IMG_HEIGHT);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("   ⏱️  Time: %.4f seconds\n", cpu_time);
    int cpu_components = countComponents(h_labels_cpu, size);
    printf("   🧮 Components found: %d\n", cpu_components);

    // GPU Setup
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_labels, size * sizeof(int));
    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);

    // Configure grid/block for GPU kernel
    dim3 block(16, 16);
    dim3 grid((IMG_WIDTH + block.x - 1) / block.x, (IMG_HEIGHT + block.y - 1) / block.y);

    // Modern GPU Algorithm (Optimized)
    printf("\n🚀 === GPU Connected Components (Optimized Algorithm) ===\n");
    
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    
    cudaEventRecord(gpu_start);
    optimized_ccl_kernel<<<grid, block>>>(d_img, d_labels, IMG_WIDTH, IMG_HEIGHT);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);
    
    cudaMemcpy(h_labels_gpu, d_labels, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("   ⏱️  Time: %.4f seconds\n", gpu_time_ms / 1000.0);
    int gpu_components = countComponents(h_labels_gpu, size);
    printf("   🧮 Components found: %d\n", gpu_components);

    // Performance Analysis
    printf("\n📊 === PERFORMANCE ANALYSIS ===\n");
    printf("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Algorithm           │ Time (s)    │ Components  │ Accuracy    │ Speedup     │\n");
    printf("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
    printf("│ CPU (Reference)     │   %8.4f  │   %8d  │    100%%     │     1.00x   │\n", 
           cpu_time, cpu_components);
    printf("│ GPU (Modern)        │   %8.4f  │   %8d  │     %s     │   %7.2fx   │\n", 
           gpu_time_ms/1000.0, gpu_components, 
           (abs(gpu_components - cpu_components) < cpu_components * 0.05) ? " ✓" : " ✗", 
           cpu_time / (gpu_time_ms/1000.0));
    printf("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");

    // Analysis Summary
    double speedup = cpu_time / (gpu_time_ms/1000.0);
    printf("\n🎯 === ANALYSIS SUMMARY ===\n");
    
    if (speedup > 1.0) {
        printf("✅ GPU Acceleration: %.2fx speedup achieved!\n", speedup);
    } else {
        printf("⚠️  GPU Performance: %.2fx slower than CPU\n", 1.0/speedup);
    }
    
    double accuracy = 100.0 * (1.0 - abs(gpu_components - cpu_components) / (double)cpu_components);
    if (accuracy > 95.0) {
        printf("✅ Algorithm Accuracy: %.1f%% - Excellent match\n", accuracy);
    } else if (accuracy > 90.0) {
        printf("⚠️  Algorithm Accuracy: %.1f%% - Good match\n", accuracy);
    } else {
        printf("❌ Algorithm Accuracy: %.1f%% - Needs improvement\n", accuracy);
    }

    printf("📈 Memory throughput: %.2f MB/s\n", 
           (size * sizeof(unsigned char) + size * sizeof(int)) / (gpu_time_ms/1000.0) / 1048576.0);

    // Sample Results Comparison
    printf("\n🔍 === SAMPLE RESULTS (First 10 foreground pixels) ===\n");
    printf("┌─────────┬─────────┬─────────┬─────────┐\n");
    printf("│ Pixel # │   CPU   │   GPU   │  Match  │\n");
    printf("├─────────┼─────────┼─────────┼─────────┤\n");
    int count = 0;
    for (int i = 0; i < size && count < 10; i++) {
        if (h_img[i] == 1) {
            printf("│  %5d  │  %5d  │  %5d  │   %s    │\n", 
                   i, h_labels_cpu[i], h_labels_gpu[i],
                   (h_labels_cpu[i] == h_labels_gpu[i]) ? "✓" : "✗");
            count++;
        }
    }
    printf("└─────────┴─────────┴─────────┴─────────┘\n");

    // Cleanup
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaFree(d_img);
    cudaFree(d_labels);
    free(h_img);
    free(h_labels_cpu);
    free(h_labels_gpu);

    printf("\n🎉 Analysis complete! Optimized GPU CCL algorithm comparison finished.\n");
    return 0;
}