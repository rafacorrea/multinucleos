#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 32

void desplegar(int *matriz, int m, int n);


__global__ void calcularGPU2D(int *mask, int *imagen, int *res, int p, int m, int n)
{
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   int j = blockIdx.y*blockDim.y + threadIdx.y;
   res[i*n+j] = 0;
   float div = (float)1/(p*p);
   int offset = p/2; //offset de la mascara
   
   //recorriendo la mascara...
   for (int k=0; k<p; k++)
   {
      for (int l=0; l<p; l++)
      {
          if ((i- offset + k)*n >=0 && (j- offset + l) >= 0)   
              res[i*n+j] += round(imagen[(i-offset + k)*n + (j-offset + l)] * div * mask[k*p+l]);
      }
   }
}

float multiplicarGPU(int *mask, int *imagen, int *res, int p, int m, int n)
{
   int *dev_mask, *dev_imagen, *dev_res;
   cudaEvent_t gpuI, gpuF;
   float gpuT;
   cudaEventCreate( &gpuI );
   cudaEventCreate( &gpuF );
   cudaEventRecord( gpuI, 0 );

   cudaMalloc( (void**)&dev_mask, p*p*sizeof(int) );
   cudaMalloc( (void**)&dev_imagen, m*n*sizeof(int) );
   cudaMalloc( (void**)&dev_res, m*n*sizeof(int) );
   cudaMemcpy( dev_mask, mask, p*p*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( dev_imagen, imagen, m*n*sizeof(int), cudaMemcpyHostToDevice );


   dim3 bloques( m/M, n/M );
   dim3 threads( M, M );
   calcularGPU2D<<<bloques, threads>>>( dev_mask, dev_imagen, dev_res, p, m, n );

   cudaDeviceSynchronize();
   cudaMemcpy( res, dev_res, m*n*sizeof(int), cudaMemcpyDeviceToHost );

   cudaEventRecord( gpuF, 0 );
   cudaEventSynchronize( gpuF );
   cudaEventElapsedTime( &gpuT, gpuI, gpuF );
   cudaFree( dev_mask );
   cudaFree( dev_imagen );
   cudaFree( dev_res );
   return gpuT;
}

void desplegar(int *matriz, int m, int n)
{
   for (int i=0; i<m; i++)
   {
      for (int j=0; j<n; j++)
         printf("%d ", matriz[i*n+j]);
      printf("\n");
   }
   printf("\n");
}

void inicializar(int *mask, int *imagen, int *res, int p, int m, int n)
{
   for(int i = 0; i<m*n; i++)
   {
      imagen[i] = 255 - ((i/n)%256);
      res[i] = 0;
   }
   for(int i = 0; i<p*p; i++)
   {
        mask[i] = (i/p) + 1;
   }
   desplegar(imagen, m, n);
}

void sumaEuler( int p, int m, int n)
{
   int *mask, *imagen, *res;
   mask = (int*) malloc(p*p*sizeof(int));
   imagen = (int*) malloc(m*n*sizeof(int));
   res = (int*) malloc(m*n*sizeof(int));

   inicializar( mask, imagen, res, p, m, n );
   
   
   printf("Tiempo (GPU): %f ms\n", multiplicarGPU( mask, imagen, res, p, m, n ) );
   
   desplegar(res, m, n);
   
   free( mask );
   free( imagen );
   free( res );
}

int main (int argc, char *argv[] )
{
    if ( argc != 4 )
    {
      printf("%s P M N\n", argv[0]);
      exit(0);
    }
    int p = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
   
    if ( p == 3 || p == 5 || p == 7 )
        if (m>n && m/n==2 && m%512 == 0 && n%256==0)
            sumaEuler (p, m, n);
    else
    {
        printf("valor incorrecto para p\n", M);
        exit(0);
    }
    return 1;
}
