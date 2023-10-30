/****************************************************************************************
An example of multiple producer multiple consumer test
The while loop performs two tasks:
1. Producing: each thread reads the source array, find elements greater than the threshold 
              value, then put the array index into the destination array
2. Consuming: each thread reads the array index from the destination array and then update
              the threashold value
Notes: A simplification here is the termination condition, because the number of elements
      greater than the threshold value is known. 
*****************************************************************************************/
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
constexpr int NUM_BLOCK = 7;
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    return;
}
__global__ void mpmc(volatile int* src, volatile int* dst, int* dst_read_counter, int* dst_write_counter, int array_size, int threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool src_process_done = false;
    int offset = 0;
    while (true) {
        if (!src_process_done) {
            int pos = 0;
            int id = tid + offset;
            if (id >= array_size) {
                src_process_done = true;
            }
            else {
                int32_t element = src[id];
                if (element > threshold) {
                    pos = atomicAdd(dst_write_counter, 1);
                    dst[pos] = id;
                    __threadfence();
                }
                offset += BLOCK_SIZE * NUM_BLOCK;
                continue;
            }
        }
        
        int old_read_val = atomicAdd(dst_read_counter, 1);
        int old_write_val = atomicAdd(dst_write_counter, 0);
        if (old_read_val >= old_write_val) {
            atomicAdd(dst_read_counter, -1);
        }
        else {
            int idx = dst[old_read_val];
            if (src[idx] > threshold) {
                src[idx] = 0;
            }
            else {
                src[idx] = 3;
            }
            dst[old_read_val] = array_size;
        }
        if (src_process_done && (old_read_val == old_write_val)) {
            break;
        }
        
    }
}


int main() {
    const int length = 1024 * 1024;
    int threshold_value = length - 108;
    int* src = new int[length];
    int* dst = new int[length];
    int valid_value = 0;
    for (int i = 0; i < length; i++) {
        src[i] = i;
        dst[i] = 0;
    }
    for (int i = length - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        swap(&src[i], &src[j]);
    }
    std::cout << " Done Initilization\n";
    int init_value = 0;
    int result;
    int* device_src;
    int* device_dst;
    int* dst_read_counter;
    int* dst_write_counter;
    cudaMalloc(&device_src, length * sizeof(int));
    cudaMalloc(&device_dst, length * sizeof(int));
    cudaMalloc(&dst_read_counter, sizeof(int));
    cudaMalloc(&dst_write_counter, sizeof(int));
    cudaMemcpy(device_src, src, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dst, dst, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_read_counter, &init_value, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_write_counter, &init_value, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mpmc << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, dst_read_counter, dst_write_counter, length, threshold_value);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_src, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result, dst_read_counter, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Run time is " << time << " ms." << std::endl;
    valid_value = length - threshold_value - 1;
    bool error = false;
    if (result != valid_value) {
        std::cout << " read result is wrong, expected: " << valid_value << " , result: " << result << std::endl;
        error = true;
    }
    cudaMemcpy(&result, dst_write_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (result != valid_value) {
        std::cout << " write result is wrong, expected: " << valid_value << " , result: " << result << std::endl;
        error = true;
    }
    
    for (int i = 0; i < length; i++) {
        if (dst[i] > threshold_value) { //% 3 != 0 ) {
            std::cout << " origin value is wrong: i " << i << " value: " << dst[i] << std::endl;
            error = true;
        }
    }
    cudaMemcpy(dst, device_dst, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < result; i++) {
        if (dst[i]!= length) {
            std::cout << " dst is wrong: i " << i << " value: " << dst[i] << std::endl;
            error = true;
        }
    }

    cudaFree(device_src);
    cudaFree(device_dst);
    cudaFree(dst_read_counter);
    cudaFree(dst_write_counter);
    delete[] src;
    delete[] dst;
    if (error) {
        return 1;
    }
    else {
        return 0;
    }
} 
