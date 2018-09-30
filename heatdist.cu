/*
 *  Please write your name and net ID below
 *  
 *  Last name: Boran
 *  First name: Tudor
 *  Net ID: N13059231
 * 
 *  I have attached a readme, you can also compile with make (which I used to get this in nsight, because I love IDEs)
 */


/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s) that you need to write too. 
 * 
 * You compile with:
 * 		nvcc -o heatdist -arch=sm_60 heatdist.cu   
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/* Tile size */
#define TILE_SIZE 8

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
__global__ void  gpu_kernel(float *, float *, unsigned int);
void  check_err(cudaError_t, char *);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements to 70F
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 70;
    
  for(i = 0; i < N; i++)
    playground[index(i,0,N)] = 70;
  
  for(i = 0; i < N; i++)
    playground[index(i,N-1, N)] = 70;
  
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 70;
  
  // from (0,10) to (0,30) inclusive are 100F
  for(i = 10; i <= 30; i++)
    playground[index(0,i,N)] = 100;
  
   // from (n-1,10) to (n-1,30) inclusive are 150F
  for(i = 10; i <= 30; i++)
    playground[index(N-1,i,N)] = 150;
  
  if( !type_of_device ) // The CPU sequential version
  {  
    start = clock();
    seq_heat_dist(playground, N, iterations);
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     gpu_heat_dist(playground, N, iterations); 
     end = clock();    
  }
  
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{


	// number of bytes to be copied between array temp and array playground
	size_t count = N*N;
	unsigned int num_bytes = count*sizeof(float);
	unsigned int i;

	float *d_temp = NULL, *d_playground = NULL;

	/* Dynamically allocate another array for temp values */
	/* Dynamically allocate NxN array of floats */
	cudaError_t err;
	err = cudaMalloc((void**)&d_temp, num_bytes);
	err = cudaMalloc((void**)&d_playground, num_bytes);
	check_err(err, "allocating memory on device.");

	err = cudaMemcpy(d_temp, playground, count*sizeof(float), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_playground, playground, count*sizeof(float), cudaMemcpyHostToDevice);
	check_err(err, "copying array to device memory.");

	dim3 block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(N/TILE_SIZE, N/TILE_SIZE, 1);
	for (i = 0; i < iterations; i++){
		gpu_kernel<<<grid, block>>>(d_playground, d_temp, N);
		err = cudaMemcpy(d_playground, d_temp, count*sizeof(float), cudaMemcpyDeviceToDevice);
		check_err(err, "syncing array");
	}

	err = cudaMemcpy(playground, d_playground, count*sizeof(float), cudaMemcpyDeviceToHost);
	check_err(err, "copying array back to host.");
}

__global__
void gpu_kernel(float *d_playground, float *d_temp, unsigned int N)
{
	unsigned int upper = N;
	unsigned int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i > 0 && i < upper && j > 0 && j < upper)
	{
		d_temp[index(i,j,N)] = (d_playground[index(i-1,j,N)] +
				d_playground[index(i+1,j,N)] +
				d_playground[index(i,j-1,N)] +
				d_playground[index(i,j+1,N)])/4.0;
	}
}

void  check_err(cudaError_t err, char *msg)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s\n", msg);
		exit(1);
	}
}


