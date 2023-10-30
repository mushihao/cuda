// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

constexpr int BLOCK_SIZE = 32;
constexpr int NUM_BLOCK = 92 * 1024;
__global__ void atomic_add_throughput(uint32_t* src1, int array_size, int iter, uint32_t mask) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < iter; ++j) {
        for (int i = tid; i < array_size; i += NUM_BLOCK * BLOCK_SIZE) {
            atomicAdd(src1 + (i & mask), 1);
        }
    }

}

__global__ void atomic_cas_throughput(uint32_t* src1, int array_size, int iter, uint32_t mask) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < iter; ++j) {
        for (int i = tid; i < array_size; i += NUM_BLOCK * BLOCK_SIZE) {
            atomicCAS(src1 + (i & mask), 1, 0);
        }
    }

}

__global__ void np_atomic_add_throughput(uint32_t* src1, int array_size, int iter, uint32_t mask) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < iter; ++j) {
        for (int i = tid; i < array_size; i += NUM_BLOCK * BLOCK_SIZE) {
            uint32_t tmp = atomicAdd(src1 + (i & mask), 1);
            if (tmp > 10000) {
                src1[i] = 0;
            }
        }
    }

}

__global__ void np_atomic_cas_throughput(uint32_t* src1, int array_size, int iter, uint32_t mask) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < iter; ++j) {
        for (int i = tid; i < array_size; i += NUM_BLOCK * BLOCK_SIZE) {
            uint32_t tmp = atomicCAS(src1 + (i & mask), 1, 0);
            if (tmp > 10000) {
                src1[i] = 0;
            }
        }
    }

}

int main() {
    const int length = 46 * 1024 * 1024;
    uint32_t* src1 = new uint32_t[length];
    for (int i = 0; i < length; i++) {
        src1[i] = 0;
    }
    uint32_t* device_src1;
    int iter = 1000 ;
    cudaMalloc(&device_src1, length * sizeof(int));
    cudaMemcpy(device_src1, src1, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    atomic_add_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0xFFFFFFFF);
    //atomic_cas_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0xFFFFFFFF);
    //np_atomic_add_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0xFFFFFFFF);
    //np_atomic_cas_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0xFFFFFFFF);
    // same address atomic
    //iter = 2;
    //atomic_add_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0x0);
    //atomic_cas_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0x0);
    //np_atomic_add_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0x0);
    //np_atomic_cas_throughput << <NUM_BLOCK, BLOCK_SIZE >> > (device_src1, length, iter, 0x0);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Run time is " << time << " ms." << std::endl;
    std::cout << "Throughput is " << length * sizeof(uint32_t) * 1.0 * iter/ 1024 / 1024 / 1024 / (time / 1000) << " GB/s" << std::endl;
    
    cudaFree(device_src1);
    delete[] src1;
    return 0;
}
