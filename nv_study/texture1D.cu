#include <string.h>
#include <math.h>
#include <vector>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

constexpr int BLOCK_SIZE = 128;
constexpr int NUM_BLOCK = 512;
constexpr int kIter = 20;

// Normal Load bandwidth
__global__ void bandwidth_test(volatile int* src, long long int* dst, int length) {
    int tid = threadIdx.x;// +blockIdx.x * blockDim.x;
    int result = 0;
    long long int start_time, end_time;
    start_time = clock64();
    #pragma unroll
    for (int j = 0; j < kIter; ++j) {
        for (int i = 0; i < length; i = i + blockDim.x) {
            result += src[tid + i];
        }
    }
    __syncthreads();
    end_time = clock64();
    if (threadIdx.x == 0) {
        dst[2047] = result;
        dst[blockIdx.x * 2] = start_time;
        dst[blockIdx.x * 2 + 1] = end_time;
    }

}

// Texture Load bandwidth
__global__ void texture_bandwidth_test(volatile int* src, long long int* dst, int length, cudaTextureObject_t texObj) {
    int tid = threadIdx.x;// +blockIdx.x * blockDim.x;
    int result = 0;
    long long int start_time, end_time;
    start_time = clock64();
#pragma unroll
    for (int j = 0; j < kIter; ++j) {
        for (int i = 0; i < length; i = i + blockDim.x) {
            result += tex1Dfetch<int>(texObj, i + tid);
        }
    }
    __syncthreads();
    end_time = clock64();
    if (threadIdx.x == 0) {
        dst[2047] = result;
        dst[blockIdx.x * 2] = start_time;
        dst[blockIdx.x * 2 + 1] = end_time;
    }

}

int main() {
    const int length = 512 * 1024;
    const int loop_cnt = 20;
    int* src = new int[length];
    long long int* dst = new long long int[2048];
    int valid_value = 0;
    for (int i = 0; i < length; i++) {
        src[i] = i;
    }
   
    std::cout << " Done Initilization\n";
    int* device_src;
    long long int* device_dst;
    cudaMalloc(&device_src, length * sizeof(int));
    cudaMalloc(&device_dst, 2048 * sizeof(long long int));
    cudaMemcpy(device_src, src, length * sizeof(int), cudaMemcpyHostToDevice);
    //  cudaMemcpy(device_dst, dst, length * sizeof(int), cudaMemcpyHostToDevice);
    
    // CUDA Texture
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = device_src;
    resDesc.res.linear.sizeInBytes = length * sizeof(int);
    resDesc.res.linear.desc = cudaCreateChannelDesc<unsigned>();

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);



    texture_bandwidth_test << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, length, texObj);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < loop_cnt; ++i) {
        texture_bandwidth_test << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, length, texObj);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_dst, 2048 * sizeof(long long int), cudaMemcpyDeviceToHost);
    printf("Total Bandwidth is %.3f GB/s\n", length * sizeof(int) * loop_cnt * kIter * NUM_BLOCK / (1.024 * 1.024 * 1.024 * time * 1000000));
    long long int sum_time = 0;
    for (int i = 0; i < NUM_BLOCK; ++i) {
        sum_time += dst[i * 2 + 1] - dst[i * 2];
    }
    printf("Total Cycles %u, avg cycles per block %.3f, averge Bandwidth per SM %.3f bytes/clk\n", sum_time, sum_time * 1.0 / NUM_BLOCK, length * sizeof(int) * loop_cnt * kIter * NUM_BLOCK * 1.0 / sum_time);
    printf("Result is %u\n", dst[2047]);
    // Destroy texture object
    cudaDestroyTextureObject(texObj);
    cudaFree(device_src);
    cudaFree(device_dst);
    delete[] src;
    delete[] dst;
    return 0;
}
