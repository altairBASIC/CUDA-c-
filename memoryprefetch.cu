#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        a[i] = num;
    }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *vector, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (vector[i] != target)
        {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
            exit(1);
        }
    }
    printf("Success! All values calculated correctly.\n");
}

int main()
{
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 2<<24; // Large number of elements
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 32 * numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    /*
     * Prefetch the memory to the device.
     * Experiment 1: Prefetch one vector (e.g., vector a)
     * Uncomment the lines below for each experiment.
     */
    cudaMemPrefetchAsync(a, size, deviceId);
    // cudaMemPrefetchAsync(b, size, deviceId);
    // cudaMemPrefetchAsync(c, size, deviceId);

    /*
     * Experiment 2: Prefetch two vectors (e.g., vectors a and b)
     * Uncomment the lines below for this experiment.
     */
    // cudaMemPrefetchAsync(a, size, deviceId);
    // cudaMemPrefetchAsync(b, size, deviceId);
    
    /*
     * Experiment 3: Prefetch all three vectors (a, b, and c)
     * Uncomment the lines below for this experiment.
     */
    // cudaMemPrefetchAsync(a, size, deviceId);
    // cudaMemPrefetchAsync(b, size, deviceId);
    // cudaMemPrefetchAsync(c, size, deviceId);

    /*
     * Launch kernels to initialize the vectors in parallel on the GPU.
     */
    initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);

    /*
     * Synchronize to ensure kernels are done.
     */
    cudaDeviceSynchronize();

    /*
     * Perform vector addition on GPU.
     */
    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    addVectorsErr = cudaGetLastError();
    if (addVectorsErr != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if (asyncErr != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(asyncErr));

    /*
     * Check if all elements are correctly calculated.
     */
    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
