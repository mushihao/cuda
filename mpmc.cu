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
constexpr int NUM_BLOCK = 64;
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    return;
}
__global__ void mpmc(volatile int* src, volatile int* dst, int* dst_read_batch_offset, int* dst_write_counter, int* processed_num, int array_size, int threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool src_process_done = false;
    int offset = 0;
    bool batch_offset_assigned = false;
    __shared__ int batch_offset;
    __shared__ unsigned int group_mask;
    if (threadIdx.x == 0) {
        batch_offset = 0;
        group_mask = 0;
    }
    __syncthreads();
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
                    // update value first, then update total processed count, to ensure memory consistency
                    atomicAdd(processed_num, 1);
                }
                offset += BLOCK_SIZE * NUM_BLOCK;
                continue;
            }
        }
        
        if (threadIdx.x == 0) {
            if (!batch_offset_assigned) {
                int batch_id = atomicAdd(dst_read_batch_offset, 1);
                batch_offset_assigned = true;
                batch_offset = batch_id * BLOCK_SIZE;               
            }
        }
        __syncthreads();
        int current_processed_num = *processed_num;
        
        if ((group_mask & (1 << threadIdx.x)) == 0) {
            if (current_processed_num > batch_offset + threadIdx.x) {
                int idx = dst[batch_offset + threadIdx.x];
                if (src[idx] > threshold) {
                    src[idx] = 0;
                }
                else {
                    src[idx] = 0x5555AAAA;
                }
                dst[batch_offset + threadIdx.x] = array_size;
                atomicOr(&group_mask, 1 << threadIdx.x); 
                //printf("group_mask is %x\n", group_mask);
            }
        }
        if (threadIdx.x == 0) {
            if (__popc(group_mask) == BLOCK_SIZE) {
                batch_offset += BLOCK_SIZE * NUM_BLOCK;
                group_mask = 0;
                //printf("batch offset is %d, processed_num %d\n", batch_offset, *processed_num);
            }
        }
        __syncthreads();
        if (batch_offset + __popc(group_mask) >= (array_size - 1 - threshold)) {
            break;
        }
    }
}


int main() {
    const int length = 8 * 1024 * 1024;
    const int loop_cnt = 1;
    int threshold_value = length * 3 / 4;
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
    int* processed_count;
    cudaMalloc(&device_src, length * sizeof(int));
    cudaMalloc(&device_dst, length * sizeof(int));
    cudaMalloc(&dst_read_counter, sizeof(int));
    cudaMalloc(&dst_write_counter, sizeof(int));
    cudaMalloc(&processed_count, sizeof(int));
    cudaMemcpy(device_src, src, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dst, dst, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_read_counter, &init_value, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_write_counter, &init_value, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(processed_count, &init_value, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mpmc << <NUM_BLOCK, BLOCK_SIZE >> > (device_src, device_dst, dst_read_counter, dst_write_counter, processed_count, length, threshold_value);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dst, device_src, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result, dst_read_counter, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Run time is " << time / loop_cnt << " ms." << std::endl;
    valid_value = length - threshold_value - 1;
    bool error = false;
    if (result != NUM_BLOCK) {
        std::cout << " block id read result is wrong, expected: " << valid_value << " , result: " << result << std::endl;
        error = true;
    }
    cudaMemcpy(&result, dst_write_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (result != valid_value) {
        std::cout << " processed write result is wrong, expected: " << valid_value << " , result: " << result << std::endl;
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
    cudaFree(processed_count);
    delete[] src;
    delete[] dst;
    if (error) {
        return 1;
    }
    else {
        return 0;
    }
} 
