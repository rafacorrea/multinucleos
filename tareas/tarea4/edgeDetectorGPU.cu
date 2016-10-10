#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>
#include<math.h>

using namespace std;
using namespace cv;

#define Q 32


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void convolucionGPU(int *mask1, int* mask2, unsigned char *img, unsigned char *res, int M, int N)
{
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   int j = blockIdx.y*blockDim.y + threadIdx.y;
   
   
   i += 1;
   j += 1;
   
   if (i*M+j >= M*N)
    return;
   
   int temp1 = ( img[i*M+j] * mask1[4] + img[(i-1)*M+j-1] * mask1[0] + img[(i-1)*M+j] * mask1[1] + img[(i-1)*M+j+1] * mask1[2] + img[i*M+j-1] * mask1[3] + img[i*M+j+1] * mask1[5] + img[(i+1)*M+j-1] * mask1[6] + img[(i+1)*M+j] * mask1[7] + img[(i+1)*M+j+1] * mask1[8] ) / 9;
   
   int temp2 = ( img[i*M+j] * mask2[4] + img[(i-1)*M+j-1] * mask2[0] + img[(i-1)*M+j] * mask2[1] + img[(i-1)*M+j+1] * mask2[2] + img[i*M+j-1] * mask2[3] + img[i*M+j+1] * mask2[5] + img[(i+1)*M+j-1] * mask2[6] + img[(i+1)*M+j] * mask2[7] + img[(i+1)*M+j+1] * mask2[8] ) / 9;
   
   int sum = abs(temp1) + abs(temp2);
   sum = sum > 255 ? 255 : sum;
   sum = sum < 0 ? 0 : sum;
   res[i*M+j] = static_cast<unsigned char> (sum);
   //if ( i == 1 && j == 1)
    //printf ("%d\n", res[i*M+j]);
   
}


void prewittEdge(const Mat &input, Mat &output)
{
    size_t inputBytes = input.step * input.rows;
    size_t outputBytes = output.step * output.rows;
    size_t maskBytes = 9*sizeof(int);
    unsigned char *d_input, *d_output;
    int *d_mask1, *d_mask2;
    
    int mask1[9] = {-1,0,1,-1,0,1,-1,0,1};
    int mask2[9] = {-1,-1,-1,0,0,0,1,1,1};

    cudaEvent_t cpuI, cpuF;
    float cpuT;
    cudaEventCreate( &cpuI );
    cudaEventCreate( &cpuF );
    cudaEventRecord( cpuI, 0 );
   
    
	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc ( (void**)&d_mask1, maskBytes ), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc ( (void**)&d_mask2, maskBytes ), "CUDA Malloc Failed");
	
	
	SAFE_CALL(cudaMemcpy( d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice ), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy( d_mask1, mask1, 9*sizeof(int), cudaMemcpyHostToDevice ), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy( d_mask2, mask2, 9*sizeof(int), cudaMemcpyHostToDevice ), "CUDA Memcpy Host To Device Failed");
	
	dim3 bloques((input.rows - 2)/Q + 1, (input.cols - 2)/Q + 1);
    dim3 threads(Q, Q);
    
	convolucionGPU<<<bloques,threads>>>(d_mask1, d_mask2, d_input, d_output, input.rows, input.cols);
	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    cudaEventRecord( cpuF, 0 );
    cudaEventSynchronize( cpuF );
    cudaEventElapsedTime( &cpuT, cpuI, cpuF);
    
    printf("Tiempo %f: ", cpuT);
    
	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

}

int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "test.JPG";
	else
  	    imagePath = argv[1];
  	
	//Read input image from the disk in greyscale
	Mat input = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	//Mat output(input.rows, input.cols, CV_8UC1);


	//Allow the windows to resize
	namedWindow("Input", WINDOW_NORMAL);


    Mat output;
    output = input.clone(); //hagamos un clon
    for(int y = 0; y < input.rows; y++) //recorramos las filas
        for(int x = 0; x < input.cols; x++) //recorramos las columnas
          output.at<uchar>(y,x) = 0; //punto inicial
    
    
    prewittEdge(input, output);
    

	//Show the input and output
	namedWindow("Input", WINDOW_NORMAL);
	imshow("Input", input);
	namedWindow("Output", WINDOW_NORMAL);
	imshow("Output", output);

	//Wait for key press
	waitKey();

	return 0;
}
