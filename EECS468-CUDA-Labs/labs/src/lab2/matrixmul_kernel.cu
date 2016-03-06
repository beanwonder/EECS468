/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    const int TILE_DIM = 32;
    __shared__ float Ms[TILE_DIM][TILE_DIM+1];
    __shared__ float Ns[TILE_DIM][TILE_DIM+1];

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

#endif // #ifndef _MATRIXMUL_KERNEL_H_
