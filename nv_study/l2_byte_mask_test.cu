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
constexpr int NUM_BLOCK = 2;
__global__ void l2_byte_mask(volatile int* src, volatile int* dst) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        for (int i = 0; i < 32; i++) {  // 32 write
            dst[i] = 0xffff;
        }
        
    }
    if (tid != 0) {
        for (int i = 1024; i < 4096; i = i + 32) { // 96 read + 96 write
            dst[i] = src[i];
        }
        dst[1] = 0xabc;  //1 write 
         
        int a = dst[0] + dst[32]; // 2 read
        if (a == 0xffff) {  // 2 write
            dst[32] = 0xabcd;
            dst[64] = 0xabcde;
        }
        
        
    }
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
    l2_byte_mask<< <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_dst, length * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Run time is " << time / loop_cnt << " ms." << std::endl;
    if (dst[1] != 0xabc || dst[32] != 0xabcd || dst[64] != 0xabcde) {
        std::cout << dst[0] << " " << dst[1] << " " << dst[31] << " " << dst[32] << " error!\n";
    }
    cudaFree(device_src);
    cudaFree(device_dst);
    delete[] src;
    delete[] dst;
    return 0;
}
