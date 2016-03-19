#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Lab4: Host Helper Functions (allocate your own data structure...)
float* allocateDeviceArr(int numElements) {
    int size = numElements * sizeof(float);
    float *dev = NULL;
    cudaMalloc((void**)&dev, size);
    return dev;
}

float* allocateArr(int numElements) {
    // float *host = NULL;
    return (float*) malloc(numElements*sizeof(float));
}

void copy2DeviceArr(float *deviceArr, const float *hostArr, int numElements) {
    int size = numElements * sizeof(float);
    cudaMemcpy(deviceArr, hostArr, size, cudaMemcpyHostToDevice);
}

void copyFromDeviceArr(float *hostArr, const float *devArr, int numElements) {
    int size = numElements * sizeof(float);
    cudaMemcpy(hostArr, devArr, size, cudaMemcpyDeviceToHost);
}


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ recudtionKernel(float *scanArr, int numElements) {
  // step 1
  int stride = 1;
  while (stride < BLOCK_SIZE) {
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx < BLOCK_SIZE) {
      scanArr[idx] += scanArr[idx-stride];
    }
    stride *= 2;
  }
  __syncthreads();
}

__global__ postScanKernenl(float *scanArr, int numElements) {
  // step 2
  int stride = BLOCK_SIZE >> 1;
  while (stride > 1) {
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx < BLOCK_SIZE) {
      scanArr[idx+stride] += scanArr[idx];
    }
    stride >>= 1;
    __syncthreads();
  }
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
  float *devArr = allocateDeviceArr(numElements);
  copy2DeviceArr(inArray, devArr, numElements);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(1);

  recudtionKernel<<<dimGrid, dimBlock>>>(devArr, numElements);
  postScanKernenl<<<dimGrid, dimBlock>>>(devArr, numElements);

  copyFromDeviceArr(outArray, devArr, numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
