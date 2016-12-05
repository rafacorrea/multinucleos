#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define WARP_SIZE 32
#define MAX_THREADS_X 1024
#define MAX_THREADS_Y 1024
#define MAX_THREADS_Z 64
#define MAX_BLOCKS_X 2147483647
#define MAX_BLOCKS_Y 65535
#define MAX_BLOCKS_Z 65535

#define THREADS_PER_BLOCK 128
//3.0, 16 blocks, 2048 threads
//MIN THREADS_PER_BLOCK = 128

#define SIZE 1024
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		//cin.get();
		exit(EXIT_FAILURE);
	}
}


#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


__global__ void scan(int *input, int n, int d, int size)  
{     
	int i = blockIdx.x *blockDim.x + threadIdx.x;
    //printf("i: %d\n", i);
    //if(n == 1)
    //    input[i + (int)pow(2.0, (double)(d+1)) - 1] = 0;
    //else
        
        //if (i < n )
        {
            i*=pow(2.0, (double)d+1);
            if(i + (int)pow(2.0, (double)(d+1)) - 1 < size)
            {
                
                input[i + (int)pow(2.0, (double)(d+1)) - 1] = input[i + (int)pow(2.0,(double)d) - 1] + input[i + (int)pow(2.0,(double)(d + 1)) - 1];
                //printf("[%d(%d+%d-1)] = [%d] + [%d], SIZE=%d\n", i + (int)pow(2.0, (double)(d+1)) - 1, i,(int)pow(2.0, (double)(d+1)),  i + (int)pow(2.0,(double)d) - 1,i + (int)pow(2.0,(double)(d + 1)) - 1, size);
            }
        }
    
    
}

__global__ void down_sweep(int *input, int n, int d, int size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
    //if (i < n)
    {
        i*=pow(2.0, (double)d+1);
        if(i + (int)pow(2.0, (double)(d+1)) - 1 < size)
        {
            int temp = input[i + (int)pow(2.0, (double)d) - 1];
            input[i + (int)pow(2.0, (double)d) -1] = input[i + (int)pow(2.0, (double)(d+1)) - 1];
            input[i +  (int)pow(2.0, (double)(d+1)) - 1] = temp + input[i +  (int)pow(2.0, (double)(d+1)) - 1];
        }
    }
}

__global__ void quickfix(int *input, int size)
{
    input[size-1] = 0;
}

int main()
{
    int * input = (int *)malloc(SIZE * sizeof(int));// = {1,2,3,4,5,6,7,8,9};
    for (int i = 0; i < SIZE; i++)
        input[i] = i;
    int *d_input;
    int d = ceil(log2((float)SIZE));
    SAFE_CALL(cudaMalloc<int>(&d_input, SIZE*sizeof(int)), "CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_input,  input, SIZE*sizeof(int), cudaMemcpyHostToDevice ), "CUDA Memcpy Host To Device Failed");
    for (int i = 0; i < d; i++)
    {    
        //int numop = ceil(SIZE/2)  
        int numop = ceil(SIZE/pow(2, i+1));
        int bloques = ceil((float)numop/THREADS_PER_BLOCK);
        printf("numop: %d", numop);
        scan<<<bloques,THREADS_PER_BLOCK>>>(d_input, numop, i, SIZE);
    }
    
    quickfix<<<1,1>>>(d_input, SIZE);
    int numop2 = 1;
    for (int i = d - 1; i >= 0; i--)
    {
      
        int bloques = ceil((float)numop2/THREADS_PER_BLOCK);
        down_sweep<<<bloques,THREADS_PER_BLOCK>>>(d_input, numop2, i, SIZE);
        numop2*=2;
    }
    SAFE_CALL(cudaMemcpy(input, d_input, SIZE*sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
    SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
    
    for(int i = 0; i < SIZE; i++)
    {
        printf("%d\n", input[i]);
    }
    
    free(input);
}
