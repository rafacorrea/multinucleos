#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

#define N 3
#define M 3

using namespace std;
using namespace cv;



void sumaEulerCPU1(int *a, int *b, int *c)
{
   // Se calculan valores sin considerar la orilla para 3x3.
   for (int i=1; i<N-1; i++)
      for (int j=1; j<M-1; j++)
         c[i*M+j] = ( b[i*M+j] * a[4] + b[(i-1)*M+j-1] * a[0] + b[(i-1)*M+j] * a[1] + b[(i-1)*M+j+1] * a[2] + b[i*M+j-1] * a[3] + b[i*M+j+1] * a[5] + b[(i+1)*M+j-1] * a[6] + b[(i+1)*M+j] * a[7] + b[(i+1)*M+j+1] * a[8] ) / 9;
         // En medio, esq sup izq, arriba, esq sup der, izq, der, esq inf izq, abajo, esq inf der
}


float convolucionCPU1(int *a, int *b, int *c)
{
   cudaEvent_t cpuI, cpuF;
   float cpuT;
   cudaEventCreate( &cpuI );
   cudaEventCreate( &cpuF );
   cudaEventRecord( cpuI, 0 );

   sumaEulerCPU1(a, b, c);

   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF );
   return cpuT;
}




int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "space-wallpaper_2880x1800.jpg";
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

	/*int *masPxP, *imaMxN, *resMxN;
   	masPxP = (int*) malloc(P*P*sizeof(int));
        imaMxN = (int*) malloc(M*N*sizeof(int));
        resMxN = (int*) malloc(M*N*sizeof(int));
	*/
	int arregloX[9] = {-1,0,1,-1,0,1,-1,0,1};

	int arregloY[9] = {-1,-1,-1,0,0,0,1,1,1};

	int arregloP[9];


	int n=3;
	/*Multiplicación de matrices*/
	 for (int i = 0; i<n; i++)
	    {
		for (int j = 0; j<n; j++)
		{
		   arregloP[i*n+j]=0;

		    for (int k = 0; k < n; k++)
		    {
	 		arregloP[i*n+j] = (arregloP[i*n+j]) + (arregloX[i*n + k] * arregloY[k*n + j]);
		    }
		}
	    }

	for(int i=0; i<3*3; ++i)
	{
	 if(i%3 == 0)
	 printf("\n");
	 printf("%d", arregloP[i]);
	}
	

	
	//convolucionCPU1( arregloX, input , arregloP );



        /*Imprimir el valor de la matriz*/	
/*

	printf("Rows:  %d", input.rows);
	uint8_t *myData = input.data;

	int _stride = input.step;//in case cols != strides
	for(int i=0; i <input.rows; ++i)
	for(int j=0; j < input.cols; ++j)
	printf("%d  ,  ", myData[ i * _stride + j]);
*/




	//Show the input and output
	imshow("Input", input);



	//Wait for key press
	waitKey();

	return 0;
}
