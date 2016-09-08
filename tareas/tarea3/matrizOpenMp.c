#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ROWS1 4
#define COLS1 3
#define ROWS2 3
#define COLS2 5

int * multiplicar (int * mat1, int *mat2, int n)
{
    
    int * res;
    res = malloc(n * n * sizeof(int));
    int temp;

    #pragma omp parallel for schedule(static)     
    for (int i = 0; i<n; i++)
    {
        for (int j = 0; j<n; j++)
        {

            temp = 0;
            for (int k = 0; k < n; k++)
            {
                temp += mat1[i*n + k] * mat2[k*n + j];
                //printf("%d * %d\n", mat1[i*COLS1 + k], mat2[k*COLS2 + j]);
            }
            res[i*n+j] = temp;
            //printf("i: %d, j:%d, temp: %d\n", i,j,temp);
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
        mat1 = malloc(n * n * sizeof(int));
        mat2 = malloc(n * n * sizeof(int));
        srand (time(NULL));
        

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
        
        printM(mat1, n, n);
        printM(mat2, n, n);
        
        int * res = multiplicar(mat1, mat2, n);

        if (res !=0)
        {
	    printf("\nMatriz de Resultado\n\n");
            printM(res, n, n);
        }
        
        return 0;
    }
}


