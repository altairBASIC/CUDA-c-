#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

__global__ void saxpy(int *a, int *b, int *c) {
    // Compute global thread ID to assign unique work to each thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use strided loop to process multiple elements in case N > total number of threads
    for (int i = tid; i < N; i += stride) {
        c[i] = 2 * a[i] + b[i];
    }
}

int main() {
    int *a, *b, *c;
    int size = N * sizeof(int);  // Calculate total memory size for each vector

    int deviceId;
    int numberOfSMs;

    // Get device ID and the number of Streaming Multiprocessors (SMs) for optimal block configuration
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Allocate unified memory for vectors a, b, and c
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize vectors a and b on the host
    for (int i = 0; i < N; ++i) {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }

    // Prefetch memory to the device (GPU)
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int number_of_blocks = numberOfSMs * 32;  // Create enough blocks to maximize SM occupancy

    // Launch the SAXPY kernel
    saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c);

    // Synchronize to ensure all threads have finished before proceeding
    cudaDeviceSynchronize();

    // Print the first and last 5 elements of vector c for a quality check
    for (int i = 0; i < 5; ++i)
        printf("c[%d] = %d, ", i, c[i]);
    printf("\n");
    for (int i = N - 5; i < N; ++i)
        printf("c[%d] = %d, ", i, c[i]);
    printf("\n");

    // Free the allocated unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
