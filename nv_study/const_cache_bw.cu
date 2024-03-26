#include <stdio.h>

#define CARRAY_SIZE 16384

__constant__ unsigned int d_carray[CARRAY_SIZE];
unsigned int h_test[CARRAY_SIZE];
unsigned int kIter = 1;

__global__ void kcbw(unsigned int* ts, unsigned int* out, int p1, int p2, int its2)
{
    int t1 = p1; int t2 = p1 * p1; int t3 = p1 * p1 + p1; int t4 = p1 * p1 + p2;
    int t5 = p1 * p2; int t6 = p1 * p2 + p1; int t7 = p1 * p2 + p2; int t8 = p2 * p1 * p2;

    int start_time, end_time;
    volatile int p;
    for (int j = 0; j < 2; j++)
    {
        p = threadIdx.x;
        int its = (j == 0) ? 2 : its2;
        start_time = clock();
            for (int k = 0; k < its; ++k) {
                
                for (int i = 0; i < CARRAY_SIZE; i = i + blockDim.x)
                {
                    t1 += d_carray[i];
                }
            }

        end_time = clock();
    }
    t1 += p;
    out[0] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

    if ((threadIdx.x  == 0))
    {
        ts[blockIdx.x * 2] = start_time;
        ts[blockIdx.x * 2 + 1] = end_time;
    }
}


void cmem_bandwidth(unsigned int* h_carray, unsigned int* d_ts, unsigned int* d_out, unsigned int* ts, unsigned int nblocks, int nthreads)
{
    dim3 Db = dim3(nthreads);
    dim3 Dg = dim3(nblocks, 1, 1);
    cudaError_t errcode;

    // Set up array contents
    for (int i = 0; i < 16384; i++)
    {
        h_carray[i] = i % 64;
        if (h_carray[i] > 16384) h_carray[i] = h_carray[i] % 16384;
    }
    cudaMemcpyToSymbol(d_carray, h_carray, CARRAY_SIZE * 4);
    unsigned long long sum_time = { 0 };
    unsigned int max_time = 0, min_time = (unsigned)-1;
    int kits = kIter;
    int its = 3;
    for (int k = 0; k < kits; k++)
    {
        kcbw << <Dg, Db >> > (d_ts, d_out, 0, 0, its);


        errcode = cudaGetLastError();
        if (errcode != cudaSuccess)
        {
            printf("Failed: %s\n", cudaGetErrorString(errcode));
        }
        cudaThreadSynchronize();
        cudaMemcpy(ts, d_ts, 4096, cudaMemcpyDeviceToHost);

        for (int p = 0; p < nblocks; ++p) {
            //printf("block ID %d, start clk %d, end clk %d\n", p, ts[p * 2], ts[p * 2 + 1]);
            if (ts[p * 2] < min_time) min_time = ts[p * 2];
            if (ts[p * 2 + 1] > max_time) max_time = ts[p * 2 + 1];
            sum_time += ts[p * 2 + 1] - ts[p * 2];
        }
    }
    
    printf("  %d: %.3f, %.3f, %.3f bytes/clk, average %.3f cycles, time span %d cycles\n", nblocks,
        (nblocks * kits * its * CARRAY_SIZE * sizeof(int) * 1.0) / sum_time,
        (nblocks * its * CARRAY_SIZE * sizeof(int) * 1.0) / max_time,
        (nblocks * its * CARRAY_SIZE * sizeof(int) * 1.0) / min_time, sum_time * 1.0 / nblocks, max_time - min_time);  // time span could have overflow issue

}

int main()
{

    unsigned int ts[4096];			// ts, output from kernel. Two elements used per thread.
    unsigned int* d_ts;
    unsigned int* d_out;			// Unused memory for storing output
    unsigned int* h_carray;

    // Allocate device array.
    cudaError_t errcode;
    if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
    {
        printf("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
        printf("   %s\n", cudaGetErrorString(errcode));
        return -1;
    }
    if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
    {
        printf("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
        return -1;
    }

    h_carray = (unsigned int*)malloc(CARRAY_SIZE * 4);

    printf("Constant cache bandwidth\n");
    for (int nblocks = 1; nblocks <= 64; nblocks++)
        cmem_bandwidth(h_carray, d_ts, d_out, ts, nblocks, 64);
    printf("\n");
    
    cudaFree(d_ts);
    cudaFree(d_out);
    free(h_carray);

    return 0;
}
    
