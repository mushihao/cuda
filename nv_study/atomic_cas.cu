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

constexpr int BLOCK_SIZE = 32;
constexpr int NUM_BLOCK = 2;
template<typename T>
__global__ void test_kernel(T* src, volatile int* dst) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicCAS(src + tid, 1, 1);
}


int main() {
    const int length = 16 * 1024;
    const int loop_cnt = 1;
    int* src = new int[length];
    int* dst = new int[length];
    int valid_value = 0;
    for (int i = 0; i < length; i++) {
        src[i] = i;
        dst[i] = 0;
    }

    std::cout << " Done Initilization\n";
    int* device_src;
    int* device_dst;
    cudaMalloc(&device_src, length * sizeof(int));
    cudaMalloc(&device_dst, length * sizeof(int));
    cudaMemcpy(device_src, src, length * sizeof(int), cudaMemcpyHostToDevice);
    //  cudaMemcpy(device_dst, dst, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_kernel<int> << <NUM_BLOCK, BLOCK_SIZE >> > (reinterpret_cast<int*>(device_src), device_dst);
    test_kernel<unsigned long long int> << <NUM_BLOCK, BLOCK_SIZE >> > (reinterpret_cast<unsigned long long int*>(device_src), device_dst);
    test_kernel<unsigned short> << <NUM_BLOCK, BLOCK_SIZE >> > (reinterpret_cast<unsigned short*>(device_src), device_dst);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_dst, length * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Run time is " << time / loop_cnt << " ms." << std::endl;
    cudaFree(device_src);
    cudaFree(device_dst);
    delete[] src;
    delete[] dst;
    return 0;
}
