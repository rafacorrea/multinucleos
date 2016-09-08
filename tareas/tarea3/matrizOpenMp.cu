#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


int * multiplicar (int * mat1, int *mat2, int n)
{
    
    int * res;
    res = (int*) malloc(n * n * sizeof(int));
    int temp; int i = 0; int j = 0; int k = 0;
//#pragma omp parallel 
    //#pragma omp parallel for schedule(static) private (i,j, k)//collapse(2)
 
 	#pragma omp parallel for ordered

    for (i = 0; i<n; i++)
    {
	#pragma omp ordered
//printf("%d, ", i);
        for (j = 0; j<n; j++)
        {

            temp = 0;
//#pragma omp critical
            for (k = 0; k < n; k++)
            {
                temp += mat1[i*n + k] * mat2[k*n + j];

            }

            res[i*n+j] = temp;

        }

    }
    
    return res;
}

void printM(int * data, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", data[i*cols+j]);
            
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
        int *mat1;
        int *mat2;
        mat1 = (int*) malloc(n * n * sizeof(int));
        mat2 = (int*) malloc(n * n * sizeof(int));
        srand (time(NULL));

   	cudaEvent_t inicio, fin;
   	float tiempo;
        

        for(int i = 0; i<n*n; i++)
        {
           // mat1[i] = rand()%991 + 10;
		mat1[i] = i;
        }
        
        for(int i = 0; i<n*n; i++)
        {
           // mat2[i] = rand()%991 + 10;
		mat2[i] = i;
            
        }
        
       /* printM(mat1, n, n);
        printM(mat2, n, n);
	*/

   cudaEventCreate( &inicio );
   cudaEventCreate( &fin );
   cudaEventRecord( inicio, 0 );
        
        int * res = multiplicar(mat1, mat2, n);


   cudaEventRecord( fin, 0 );
   cudaEventSynchronize( fin );
   cudaEventElapsedTime( &tiempo, inicio, fin );


        if (res !=0)
        {
	    printf("\nMatriz de Resultado\n\n");
            printM(res, n, n);
        }
	

 printf("tiempo total en ms: %f\n", tiempo);

        
        return 0;
    }
}


