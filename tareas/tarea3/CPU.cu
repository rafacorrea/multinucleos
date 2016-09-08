#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float * multiplicar (float * mat1, float *mat2, int n)
{
    
    float *res; int i=0; int j=0; int k=0;
    res =   (float*) malloc(n * n * sizeof(float));


    for (i = 0; i<n; i++)
    {

        for (j = 0; j<n; j++)
        {
	   res[i*n+j]=0;

            for (k = 0; k < n; k++)
            {

 		res[i*n+j] = (res[i*n+j]) + (mat1[i*n + k] * mat2[k*n + j]);

            }
        }

    }
    
    return res;
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

int main (int argc, char *argv[] )
{
    if ( argc != 2 ) /* argc should be 2 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s N", argv[0] );
    }
    else
    {
        int n = atoi(argv[1]);
        float *mat1;
        float *mat2;
        mat1 = (float*) malloc(n * n * sizeof(float));
        mat2 = (float*) malloc(n * n * sizeof(float));
        srand (time(NULL));

   	cudaEvent_t inicio, fin;
   	float tiempo;
        

        for(int i = 0; i<n*n; i++)
        {
            mat1[i] = rand()%991 + 10;
        }
        
        for(int i = 0; i<n*n; i++)
        {
            mat2[i] = rand()%991 + 10;
            
        }
        
       /* printM(mat1, n, n);
        printM(mat2, n, n);
	*/
   cudaEventCreate( &inicio );
   cudaEventCreate( &fin );
   cudaEventRecord( inicio, 0 );

        
        float * res = multiplicar(mat1, mat2, n);

   cudaEventRecord( fin, 0 );
   cudaEventSynchronize( fin );
   cudaEventElapsedTime( &tiempo, inicio, fin );


        if (res !=0)
        {
	        //printf("\nMatriz de Resultado\n\n");
            //printM(res, n, n);
        }


   printf("tiempo total en ms: %f\n", tiempo);


        
        return 0;
    }
}


