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
// 2D texture kernel
__global__ void transformKernel(int* output, cudaTextureObject_t texObj, int width, int height)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Read from texture and write to global memory
    output[y * width + x] = tex2D<int>(texObj, x, y);
}

__global__ void transformKernelLinear(int* output, cudaTextureObject_t texObj, int width, int height)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Read from texture and write to global memory
    output[x] = tex1Dfetch<int>(texObj, x);
}

// Host code
int main()
{
    const int height = 1024;
    const int width = 1024;

    // Allocate and set some host data
    int* h_data = (int*)std::malloc(sizeof(int) * width * height);
    for (int i = 0; i < height * width; ++i)
        h_data[i] = i;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(int);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(int),
        height, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;//cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // linear CUDA Texture
    int* device_src;
    cudaMalloc(&device_src, height * width * sizeof(int));
    cudaMemcpy(device_src, h_data, height * width * sizeof(int), cudaMemcpyHostToDevice);
    // Specify texture
    struct cudaResourceDesc resDesc_l;
    memset(&resDesc_l, 0, sizeof(resDesc_l));
    resDesc_l.resType = cudaResourceTypeLinear;
    resDesc_l.res.linear.devPtr = device_src;
    resDesc_l.res.linear.sizeInBytes = height * width * sizeof(int);
    resDesc_l.res.linear.desc = cudaCreateChannelDesc<unsigned>();

    // Specify texture object parameters
    struct cudaTextureDesc texDesc_l;
    memset(&texDesc_l, 0, sizeof(texDesc_l));
    texDesc_l.addressMode[0] = cudaAddressModeClamp;
    texDesc_l.filterMode = cudaFilterModePoint;
    texDesc_l.readMode = cudaReadModeElementType;
    texDesc_l.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj_l = 0;
    cudaCreateTextureObject(&texObj_l, &resDesc_l, &texDesc_l, NULL);

    // Allocate result of transformation in device memory
    int* output;
    cudaMalloc(&output, width * height * sizeof(int));

    // Invoke kernel
    dim3 threadsperBlock(32, 32);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
        (height + threadsperBlock.y - 1) / threadsperBlock.y);
    transformKernel << <numBlocks, threadsperBlock >> > (output, texObj, width, height);

    // Copy data from device back to host
    cudaMemcpy(h_data, output, width * height * sizeof(float),
        cudaMemcpyDeviceToHost);
    transformKernelLinear << <width * height / 256, 256 >> > (reinterpret_cast<int*>(output), texObj_l, width, height);

    // Destroy texture object
    cudaDestroyTextureObject(texObj);
    cudaDestroyTextureObject(texObj_l);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    cudaFree(device_src);

    // Free host memory
    free(h_data);

    return 0;
}
