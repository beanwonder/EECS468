#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#include "cuda.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    const int TILE_DIM = 32;
    __shared__ float Ms[TILE_DIM][TILE_DIM];
    __shared__ float Ns[TILE_DIM][TILE_DIM];

    const int Pcols = P.width;
    const int Prows = P.height;
    const int Mcols = M.width;
    const int Mrows = M.height;
    const int Ncols = N.width;
    const int Nrows = N.height;
    // calc row index
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    // calc colum index
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float Pvalue = 0.0;
    for (int k=0; k < (TILE_DIM+Mcols-1)/TILE_DIM; ++k) {
        // every thread within a block load [TILE_SIZE] into shared memery
        if (k*TILE_DIM + threadIdx.x < Mcols && row < Mrows) {
            Ms[threadIdx.y][threadIdx.x] = M.elements[row*Mcols + k*TILE_DIM + threadIdx.x];
        }
        else {
            Ms[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (k*TILE_DIM + threadIdx.y < Nrows && col < Ncols) {
            Ns[threadIdx.y][threadIdx.x] = N.elements[(k*TILE_DIM + threadIdx.y)*Ncols + col];
        } else {
            Ns[threadIdx.y][threadIdx.x] = 0.0;
        }
        // wait for load all into shared mem
        __syncthreads();

        for (int n=0; n < TILE_DIM; ++n) {
            // if ((k*TILE_DIM + n < Mcols && row < Mrows) && (k*TILE_DIM + n < Nrows && col < Ncols)) {
                // but divergence
            // no divergence since load zero for unvalid data
            Pvalue += Ms[threadIdx.y][n] * Ns[n][threadIdx.x];
            // }
        }
        __syncthreads();
    }

    if (row < Prows && col < Pcols) {
        // divergence
        // P.elements[((blockIdx.y * blockDim.y + threadIdx.y)*Pwidth)+(blockIdx.x*blockDim.x)+threadIdx.x] = Pvalue;
        P.elements[row * Pcols + col] = Pvalue;
    }
}

// matrix inversion
/*
__global__ void augmentMatrix(const Matrix M, Matrix MM) {
    // only one block do this
    const int tid = blockkIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDimx.x * gridDim.x;
    const int size = M.height * M.width;
    // const int j = blockIdx.x;
    for (int pos = tid; i < size; pos += numThreads) {
        M.elements[pos]
    }   
}
*/

/*
__device__ __global__ void MatrixInversionKernel1(Matrix Ma, Matrix Mb, int size) 
{
    // Ma Mb are int device
    int idx = threadIdx.x ; 
    int idy = threadIdx.y ; 
    
   // use share mem opt1
    __shared__ float temp[16][16]; 
    
    //data -> shared mem 
    temp[idy][idx] = Ma.elements[(idy * (size+1)) + idx] ; 
    
    for(int i =1 ; i<size ;i++) 
    { 
        if((idy + i) < size) // 
        { 
            float var1 =(-1)*( temp[i-1][i-1]/temp[i+idy][i-1]); 
            temp[i+idy][idx] = temp[i-1][idx] +((var1) * (temp[i+idy ][idx]));
        } 
        __syncthreads(); //Synchronizing all threads before Next
        // maybe slow here  
    } 
    // copy back
    Mb.elements[idy*(size+1) + idx] = temp[idy][idx]; 
}
*/

__global__ void addupKernel(Matrix M, int size, int rowId) {
    
    const int colId = threadIdx.x;
    if (M.elements[size * rowId + colID] != 0) {
        return;
    }
    
    for (int k=rowId+1; k < M.height; ++k) {
        // won't divergence
        // m[k][j] != 0 do
        if (M.elements[size * k + rowId] != 0) {
            M.elements[size * rowId + colId] += M.elements[size * k + colId];
            return;
            // printf("%f", M.elements[size * k + rowId]);
        }
    }
}

__global__ void fixRowKernel(Matrix M, int size, int rowId) {
    // !! M is a augmented matrix
    __shared__ float Ri[512];
    __shared__ float Aii;

    const int colId = threadIdx.x;
    Ri[colId] = M.elements[size * rowId + colId];
    // TODO may be wrong here
    Aii = M.elements[size * rowId + rowId];
    __syncthreads();
    
    Ri[colId] = Ri[colId] / Aii;
    M.elements[size * rowId + colId] = Ri[colId];
}

// pipline

__global__ void fixColumnKernel(Matrix M, int size, int colId) {
    
    // !! M is a augmented matrix
    const int i = threadIdx.x;
    const int j = blockIdx.x;
    
    __shared__ float col[512];
    // jth element
    __shared__ float AColIdj;
    __shared__ float colj[512];
    
    col[i] = M.elements[i * size + j];
    if (col[i] != 0) {
        colj[i] = M.elements[i * size + j];
        AColIdj = M.elements[colId * size + j];
        if (i != colId) {
            colj[i] = colj[i] - AColIdj  * col[i];
        }
        M.elements[i * M.width + j] = colj[i];
    }
}

// matrix transpose
/*
__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void copySharedMem(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}
*/
#endif // #ifndef _MATRIXMUL_KERNEL_H_
