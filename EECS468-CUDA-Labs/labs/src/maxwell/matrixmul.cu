/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
// includes, project
#include <cutil.h>

// includes, kernels
#include <matrixmul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
void MatrixInversionOnDevice(Matrix Mtemp_h , int size , Matrix Mtemp1_h);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;
	
	srand(52);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(rand() % 1024, rand() % 1024, 1);
		N  = AllocateMatrix(M.width, rand() % 1024, 1);
		P  = AllocateMatrix(M.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = NULL; //(int*)malloc(3 * sizeof(int));
		unsigned int data_read = 3;
		cutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 3){
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);		
		P  = AllocateMatrix(params[0], params[2], 0);
		errorM = ReadFile(&M, argv[2]);
		errorN = ReadFile(&N, argv[3]);
		if(errorM  || errorN )
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}

	// M * N on the device
	// MatrixMulOnDevice(M, N, P);
    Matrix res = AllocateMatrix(M.height, M.width, 0);
    MatrixInversionOnDevice(M, M.height, res);
    
	printf("GPU computation complete\n");
	// compute the matrix multiplication on the CPU for comparison
	// Matrix reference = AllocateMatrix(P.height, P.width, 0);
	// computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
        
	// printf("CPU computation complete\n");
	// in this case check if the result is equivalent to the expected soluion
	// CUTBoolean res = cutComparefe(reference.elements, P.elements, P.height*P.width, 0.001f);
	// printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    /* 
   	if(argc == 5)
	{
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
		WriteFile(P, argv[1]);
	} 
    */
    /*
    // output M
    printf("original M :\n");
    for (int i=0; i < M.height; ++i) {
        for (int j=0; j < M.width; ++j) {
            std::cout << M.elements[i * M.height + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << ("inversion M :\n");
    for (int i=0; i < res.height; ++i) {
        for (int j=0; j < res.width; ++j) {
            std::cout << res.elements[i * res.height + j] << ' ';
        }
        std::cout << '\n';
    }
    */
	// Free matrices
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);
	return 0;
}

void display_matrix(Matrix M) {
    std::cout << ("Matrix display :\n");
    for (int i=0; i < M.height; ++i) {
        for (int j=0; j < M.width; ++j) {
            std::cout << M.elements[i * M.width + j] << ' ';
        }
        std::cout << '\n';
    }
}

// MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
// this function run on cpu and calculate matrixInversion
void MatrixInversionOnDevice(Matrix Mtemp_h , int size , Matrix Mtemp1_h) 
{ 
    
    //Memory allocation on the device 
    display_matrix(Mtemp_h);
    Matrix MM_host = AllocateMatrix(Mtemp_h.height, Mtemp_h.width*2, 0);
    Matrix MM_device = AllocateDeviceMatrix(MM_host);
    
    for (int i=0; i < size; ++i) {
        for (int j=0; j < size; ++j) {
            MM_host.elements[i * MM_host.width + j] = Mtemp_h.elements[i * Mtemp_h.width + j];
        }
    }
    
    for (int i=0; i < size; ++i) {
        MM_host.elements[i * MM_host.width + size + i] = 1;
    }
    
    display_matrix(MM_host);

    CopyToDeviceMatrix(MM_device, MM_host);
    
    //Kernel call 
    // MatrixInversionKernel1<<<dimGrid, dimBlock>>>(Ma, Mb, numvar); 
    
    // MM_device augmented matrix
    dim3 dimGrid1(1);
    dim3 dimGridn(size);
    dim3 dimBlockn(size);
    for (int j=0; j < size; ++j) {
        // addup
        addupKernel<<<dimGrid1, dimBlockn>>>(MM_device, size, j);
        // step2
        fixRowKernel<<<dimGrid1, dimBlockn>>>(MM_device, size, j);
        // step3
        fixColumnKernel<<<dimGridn, dimBlockn>>>(MM_device, size, j);
    }
    
    // Coping data to host from device 
    CopyFromDeviceMatrix(MM_host, MM_device);
    //Deallocating memory on the device 
    display_matrix(MM_host);
    for (int i=0; i < size; ++i) {
        for (int j=0; j < size; ++j) {
            Mtemp1_h.elements[i * Mtemp1_h.width + j] = MM_host.elements[i * MM_host.width + size + j];
        }
    }
    FreeDeviceMatrix(&MM_device); 
    //FreeDeviceMatrix(&Mb); 
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	// Load M and N to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory

	// Setup the execution configuration
    const int TILE_DIM = 32;
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((Pd.width/TILE_DIM)+1, (Pd.height/TILE_DIM)+1);

    // Launch the device computation threads!
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height*M->width;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return (data_read != (M->height * M->width));
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    cutWriteFilef(file_name, M.elements, M.width*M.height,
                       0.0001f);
}
