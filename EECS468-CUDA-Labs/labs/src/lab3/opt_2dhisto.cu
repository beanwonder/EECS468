#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"
#include "device_function.h"
#include <cuda.h>

void opt_2dhisto(uint32_t * result, uint32_t * input, int height, int width)
{
    int size = height * width;
    dim3 dim_grid((size - 1) / 1024+1, 1, 1);
    dim3 dim_block(1024, 1, 1);

    histogram_kernel<<<dim_grid, dim_block>>>(result, input, height, width);

    cudaDeviceSynchronize();
}

uint32_t* histogram_data_device(uint32_t *input[], int h, int w)
{
    uint32_t *histo_bins = (uint32_t* ) malloc(h*w*sizeof(uint32_t));
    for(int i= 0; i < h;  i++) {
        for(int j=0; j < w; j++) {	
            histo_bins[i*w+j] = input[i][j];  
        }
    }
    return  histo_bins;
}

__global__ void histogram_kernel(uint32_t *result, uint32_t *input, int height, int width)
{
    int size = height * width;

    __shared__ uint32_t s_Hist[6][HISTO_HEIGHT * HISTO_WIDTH];

    for (size_t j = threadIdx.x; j <HISTO_HEIGHT * HISTO_WIDTH ; j += blockDim.x) {
        // segementation fo s hist gram
        s_Hist[0][j] = 0;
        s_Hist[1][j] = 0;
        s_Hist[2][j] = 0;
        s_Hist[3][j] = 0;
        s_Hist[4][j] = 0;
        s_Hist[5][j] = 0;

        if (blockIdx.x==0) {
            result[j] = 0;
            // each time you have to clear result
        }
    }
    __syncthreads();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        const uint32_t data = input[i];
        atomicAdd(s_Hist[i % 6] + data, 1);
    }

    __syncthreads();
    for (size_t i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH; i += blockDim.x) {
        const uint32_t addsup = s_Hist[0][i] + s_Hist[1][i] + s_Hist[2][i] + s_Hist[3][i] + s_Hist[4][i] + s_Hist[5][i];
        atomicAdd(result + i, addsup);
           // more concurrency exploiedk
    }
}

void* allocate_device(size_t size) 
{
    void* dev_ptr;
    cudaMalloc(&dev_ptr, size);
    return dev_ptr;
}

void copy_to_device(void* device, void* host, size_t size)
{
    cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void free_device(void *device)
{
    cudaFree(device);
}

void copy_from_device(void* host, void* device, size_t size)
{
    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

void bin32_to_bin8(uint32_t *kernel_bin32, uint8_t* kernel_bin8)
{
    for (int i =0; i < HISTO_HEIGHT*HISTO_WIDTH; ++i) {
        if (kernel_bin32[i] > 255) {
            kernel_bin8[i] = 255;
        } else {
            kernel_bin8[i] = kernel_bin32[i];
        }
    }
}


