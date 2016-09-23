#include <stdio.h>
#include <math.h>

#define N 10000000
#define THREADS_PER_BLOCK 1000

//cambia todos los numeros pares excepto el 2
__global__ void pares(char *a, int raiz)
{ 
   //calcular index que este thread revisara
   int index = blockIdx.x * blockDim.x + (threadIdx.x * 2);

   //para que se salte el 2
   if (index == 2)
       return;
   if (index < N)
      a[index] = 1;
}

//para revisar los impares
__global__ void impares(char *a, int raiz)
{
   //para que se salte el 1
   int index = blockIdx.x * blockDim.x + (threadIdx.x * 2) + 1;
   if (index == 1)
       return;

   //revisa si el numero ya fue revisado
   if (a[index] == 0)
   {
       int j;
       if (index <= raiz)
           for (j=index*index; j<N; j+=index)
               a[j] = 1;
   }
   
}

int main()
{
   //arreglo en CPU/RAM
   char *a = new char[N];

   //arreglo para el device
   char *d_a;

   //la raiz del numero maximo
   int raiz = sqrt(N);
   
   //tamanio del arreglo
   int size = N * sizeof( char );
   
   //tiempos
   float tiempo1, tiempo2;
   cudaEvent_t inicio1, fin1, inicio2, fin2; // para medir tiempos como con timestamp

   /* allocate space for host copies of a, b, c and setup input alues */

   //a = (char *)malloc( size );
  
   for( int i = 0; i < N; i++ )
      a[i] = 0;
   //cambia el 0 y el 1 (casos especiales )
   a[0] = 1;
   a[1] = 1;

   //empieza a tomar tiempo
   cudaEventCreate(&inicio1); // Se inicializan
   cudaEventCreate(&fin1);
   cudaEventRecord( inicio1, 0 ); // Se toma el tiempo de inicio

   /* allocate space for device copies of a, b, c */

   cudaMalloc( (void **) &d_a, size );


   /* copy inputs to deice */
   /* fix the parameters needed to copy data to the device */
   cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
   
   //empieza a tomar tiempo
   cudaEventCreate(&inicio2); // Se inicializan
   cudaEventCreate(&fin2);
   cudaEventRecord( inicio2, 0 ); // Se toma el tiempo de inicio

   /* launch the kernel on the GPU */
   pares<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, raiz );
   impares<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, raiz );


   cudaEventRecord( fin2, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin2 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo2, inicio2, fin2 );

   /* copy result back to host */
   /* fix the parameters needed to copy data back to the host */
   cudaMemcpy( a, d_a, size, cudaMemcpyDeviceToHost );
   
   //libera memoria
   cudaFree( d_a );

   cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin1 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo1, inicio1, fin1 );
   
   //cuenta cuantos primos hay
   int cuantos=0;
   for (int i=0; i<N; i++)
   {     
       if(a[i] == 0)
       {
           printf( "%d\n", i);
           cuantos++;
       }
       
   }
   printf( "cantidad de numeros primos: %d\n", cuantos);
 

  /* clean up */

   free(a);

   printf("Tiempo cálculo %f ms\n", tiempo2);
   printf("Tiempo total %f ms\n", tiempo1);
	
   return 0;
} /* end main */
