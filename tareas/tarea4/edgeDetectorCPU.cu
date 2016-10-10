#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>
#include<math.h>

using namespace std;
using namespace cv;

//int *b

void sumaEulerCPU1(int *a, Mat img, int *c, int N, int M)
{
   // Se calculan valores sin considerar la orilla para3x3.
   for (int i=1; i<N-1; i++)
      for (int j=1; j<M-1; j++)
         c[i*M+j] = ( img.at<uchar>(i*M+j) * a[4] + img.at<uchar>((i-1)*M+j-1) * a[0] + img.at<uchar>((i-1)*M+j) * a[1] + img.at<uchar>((i-1)*M+j+1) * a[2] + img.at<uchar>(i*M+j-1) * a[3] + img.at<uchar>(i*M+j+1) * a[5] + img.at<uchar>((i+1)*M+j-1) * a[6] + img.at<uchar>((i+1)*M+j) * a[7] + img.at<uchar>((i+1)*M+j+1) * a[8] ) / 9;
         // En medio, esq sup izq, arriba, esq sup der, izq, der, esq inf izq, abajo, esq inf der
}


float convolucionCPU1(int *a, Mat img, int *c, int N, int M)
{
   cudaEvent_t cpuI, cpuF;
   float cpuT;
   cudaEventCreate( &cpuI );
   cudaEventCreate( &cpuF );
   cudaEventRecord( cpuI, 0 );

   sumaEulerCPU1(a, img, c, N, M);

   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF);
   return cpuT;
}




int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		//imagePath = "Bikesgray.jpg";
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

	int  *resX, *resY;
   	//masPxP = (int*) malloc(P*P*sizeof(int));
        //imaMxN = (int*) malloc(M*N*sizeof(int));
        resX = (int*) malloc(input.rows*input.cols*sizeof(int));
        resY = (int*) malloc(input.rows*input.cols*sizeof(int));
	

	/*Matrices utilizadas para Prewitt*/
	int arregloX[9] = {-1,0,1,-1,0,1,-1,0,1};

	int arregloY[9] = {-1,-1,-1,0,0,0,1,1,1};

	//int arregloP[9];

	int n=3;
	/*Multiplicación de matrices*/
	/* for (int i = 0; i<n; i++)
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
	*/

	float tiempo;
	tiempo= convolucionCPU1( arregloX, input , resX, input.rows, input.cols );
        tiempo= convolucionCPU1( arregloY, input , resY, input.rows, input.cols );


	printf("Tiempo %f: ", tiempo);


	Mat final;
	final = input.clone(); //hagamos un clon
for(int y = 0; y < input.rows; y++) //recorramos las filas
    for(int x = 0; x < input.cols; x++) //recorramos las columnas
      final.at<uchar>(y,x) = 0.0; //punto inicial


float sum;

/*Se tienen las 2 matrices y se juntan para formar la imagen final en base al algoritmo de PREWITT*/


  for(int y = 1; y < input.rows - 1; y++){
    for(int x = 1; x < input.cols - 1; x++){
	//sum = sqrt ( (resX[y*input.cols+x] * resX[y*input.cols+x]) + (resY[y*input.cols+x] * resY[y*input.cols+x]) );
	//sum = sqrt(resY[y*input.cols+x]);
	//sum = sum > 255 ? 255:sum;
        //sum = sum < 0 ? 0 : sum;
     // final.at<uchar>(y,x) = sum;
      final.at<uchar>(y,x) = resX[y*input.cols+x]; // y vamos pasando los puntos.
    }
  }



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
	namedWindow("Input", WINDOW_NORMAL);
	imshow("Input", input);
	namedWindow("Output", WINDOW_NORMAL);
	imshow("Output", final);


	//Wait for key press
	waitKey();

	return 0;
}
