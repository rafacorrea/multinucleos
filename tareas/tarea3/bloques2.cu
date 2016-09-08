#include <stdio.h>
#include <stdlib.h>

//#define N 500 //tamaño de bloques

__global__ void multiplicar( float *mat1, float *mat2, float *res, int n) {

  int j=0; int k=0;
  int index = blockIdx.x;

  for (j = 0; j<n; j++)
  {
      res[index*n+j]=0;

      for (k = 0; k < n; k++)
      {
          res[index*n+j] += (mat1[index*n + k] * mat2[k*n + j]);
      }
  }
}

void printM(float * data, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.0f ", data[i*cols+j]);
            
        }
        printf("\n");
    }
    printf("\n");
}






int main( int argc, char *argv[] ) {


 if ( argc != 2 ) 

    {

        printf( "usage: %s N (N debe ser < # bloques totales grid size)", argv[0] );
    }
    else
    {
        int n = atoi(argv[1]);


   float *mat1= new float[n*n], *mat2=new float[n*n], *res=new float[n*n];
   float *mat_1, *mat_2, *mat_r;
   float tiempo1, tiempo2;
   cudaEvent_t inicio1, fin1, inicio2, fin2; // para medir tiempos como con timestamp



   cudaEventCreate(&inicio1); // Se inicializan
   cudaEventCreate(&fin1);
   cudaEventRecord( inicio1, 0 ); // Se toma el tiempo de inicio


   srand (time(NULL));
        

   // fill the arrays 'a' and 'b' on the CPU
   for (int i=0; i<n*n; i++)
      mat1[i] = mat2[i] = i;


 /* for(int i = 0; i<n*n; i++)
        {
            //mat1[i] = rand()%991 + 10;
		mat1[i] = i;
        }
        
        for(int i = 0; i<n*n; i++)
        {
            //mat2[i] = rand()%991 + 10;
		mat2[i] = i;
            
        }
*/

   // allocate the memory on the GPU
   cudaMalloc( (void**)&mat_1, n * n * sizeof(float) );
   cudaMalloc( (void**)&mat_2, n * n * sizeof(float) );
   cudaMalloc( (void**)&mat_r, n * n * sizeof(float) );

   // copy the arrays 'a' and 'b' to the GPU
   cudaMemcpy( mat_1, mat1, n * n  * sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( mat_2, mat2, n * n  * sizeof(float), cudaMemcpyHostToDevice );

   cudaEventCreate(&inicio2); // Se inicializan
   cudaEventCreate(&fin2);
   cudaEventRecord( inicio2, 0 ); // Se toma el tiempo de inicio


   multiplicar<<<n,1>>>( mat_1, mat_2, mat_r, n );

   cudaEventRecord( fin2, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin2 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo2, inicio2, fin2 );

   // copy the array 'c' back from the GPU to the CPU
   cudaMemcpy( res, mat_r, n * n  * sizeof(float), cudaMemcpyDeviceToHost );

   // free the memory allocated on the GPU
   cudaFree( mat_1 );
   cudaFree( mat_2 );
   cudaFree( mat_r );

   cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin1 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo1, inicio1, fin1 );


 	if (res !=0)
        {
	    printf("\nMatriz de Resultado\n\n");
            printM(res, n, n);
        }



   free(mat1);
   free(mat2);
   free(res);

   printf("Tiempo cálculo %f ms\n", tiempo2);
   printf("Tiempo total %f ms\n", tiempo1);

   return 0;
  }

        
        //float * res = multiplicar(mat1, mat2, n);


}//Cierre de main
