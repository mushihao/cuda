// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

constexpr int BLOCK_SIZE = 1;
constexpr int NUM_BLOCK = 1;
constexpr int kIter = 1;


__global__ void threadfence_latency_test(int* src, long long int* dst, int length) {
    int tid = threadIdx.x;// +blockIdx.x * blockDim.x;
    int idx = 0;
    long long int start_time, end_time;
    for (int i = 0; i < 128; i++) {
        dst[i] = src[i];
    }
    start_time = clock64();
    __threadfence_block();
    end_time = clock64();
    dst[idx * 2] = start_time;
    dst[idx * 2 + 1] = end_time;
    idx++;
    start_time = clock64();
    __threadfence();
    end_time = clock64();
    dst[idx * 2] = start_time;
    dst[idx * 2 + 1] = end_time;
    idx++;
    for (int i = 8; i < 1024 * 1024; i++) {
        dst[i] = src[i];
    }
    start_time = clock64();
    __threadfence_system();
    end_time = clock64();
    dst[idx * 2] = start_time;
    dst[idx * 2 + 1] = end_time;
    idx++;
}

int main() {
    const int length = 1024 * 1024;
    const int loop_cnt = 1;
    int* src = new int[length];
    long long int* dst = new long long int[length];
    int valid_value = 0;
    for (int i = 0; i < length; i++) {
        src[i] = i;
    }

    std::cout << " Done Initilization\n";
    int* device_src;
    long long int* device_dst;
    cudaMalloc(&device_src, length * sizeof(int));
    cudaMalloc(&device_dst, length * sizeof(long long int));
    cudaMemcpy(device_src, src, length * sizeof(int), cudaMemcpyHostToDevice);
    //  cudaMemcpy(device_dst, dst, length * sizeof(int), cudaMemcpyHostToDevice);

    threadfence_latency_test << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, length);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < loop_cnt; ++i) {
        threadfence_latency_test << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, length);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_dst, length * sizeof(long long int), cudaMemcpyDeviceToHost);
    unsigned long long int sum_time = 0;
    unsigned long long int min_time = 0xFFFFFFFFFFFFFFFF;
    unsigned long long int max_time = 0;
    for (int i = 0; i < 3; ++i) {
        sum_time += dst[i * 2 + 1] - dst[i * 2];
        if (min_time > dst[i * 2]) min_time = dst[i * 2];
        if (max_time < dst[i * 2 + 1]) max_time = dst[i * 2 + 1];
        printf("%u, %u, %u\n", dst[i * 2], dst[i * 2 + 1], dst[i * 2 + 1] - dst[i * 2]);
    }
   
    cudaFree(device_src);
    cudaFree(device_dst);
    delete[] src;
    delete[] dst;
    return 0;
}
