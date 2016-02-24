#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

void opt_2dhisto( /*define your own function parameters*/ )
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

}

/* Include below the implementation of any other functions you need */

uint32_t* allocate_device_histogram_bins(size_t y_size, size_t x_size) {
    // void** alloc_2d(size_t y_size, size_t x_size, size_t element_size)
    const size_t x_size_padded = (x_size + 128) & 0xFFFFFF80;
    int total_size = x_size_padded * y_size * sizeof(uint32_t);
    uint32_t *dev_ptr;
    // cudaMalloc((void**)&Mdevice.elements, size);
    cudaMalloc((void**)&dev_ptr, total_size);
    return dev_ptr;
}

uint8_t* allocate_device_bins(size_t height, size_t width) {
    int size = width * height * sizeof(uint8_t);
    uint8_t *dev_ptr;
    cudaMalloc((void**)&dev_ptr, size);
    return dev_ptr;
}

void copy_to_device_histogram_bins(uint32_t *device, const uint32_t *host, 
                                   size_t y_size, size_t x_size)
{
    const size_t x_size_padded = (x_size + 128) & 0xFFFFFF80;
	int size = x_size_padded * y_size * sizeof(uint32_t);
    cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void copy_from_device_bins(uint8_t *host, const uint8_t *device, size_t h, size_t w)
{
	int size = w * h * sizeof(uint8_t);
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}
