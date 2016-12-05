#include <stdio.h>   
#include <string.h>  
#include <stdlib.h>  
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cstring>

#include "tree_huff.cuh"

#define FILE_NAME_MAX_LEN 270
#define MAX_FILE_NAME 256
#define N 32
#define M 4

#define MAX_BLOCKS 65535
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32

void add_magic_num(FILE *file); //funcion que anade un numero magico para verificar que es un archivo generado por este proceso
void add_bit_vector(FILE *file, unsigned char bit_vector[32]); //funcion que anade al archivo la representacion de que caracteres estan presentes (isnerta 256 bits)
int  add_size(FILE *file, int freq[MAX_CHARS]);//funciona que determina el numero de bits necesarios para representar las frecuencias de los caracteres
void add_character_counts(FILE *file, int freq[MAX_CHARS], int num_bytes);//funcion que inserta header con las frecuencias de los caractres


//funcion para copiar strings en kernels
__device__ char * my_strcpy(char *dest, const char *src){
  int i = 0;
  while (src[i] != 0)
  {
    dest[i] = src[i];
    i++;
  }
  return dest;
}

//kernel que escribe a un arreglo de chars la representacion en bits de los caracteres scadaos del archivo
__global__ void encode_byte_stream(char * string, code * code_values, char * res, int * offset, int f_size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if(i < f_size)
	{
	    my_strcpy(res + offset[i], code_values[string[i]].path);
            //printf("hola\n");
	}
	
	
}

//kernel que compacta 8 caracteres (0 o 1) en un caracter o byte
__global__ void compressed_bit_stream(char * encoded_byte_stream, unsigned char * res, int f_size)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < f_size)
	{
	    unsigned char packed = 0;
        for (int count = 0; count < 8; count++) {
           if (encoded_byte_stream[(i*8)+count] == '0') {
              packed <<= 1;
           } else {
              packed = (packed << 1) | 0x01;
           }
         }
         
	     res[i] = packed;
	}
}
int main(int argc, char *argv[]) {


   //archivos 
   FILE *file_in, *file_out;
   int freq[MAX_CHARS] = {0}, c = 0, count = 0, num_bytes = 0, ret = 0;
   unsigned char bit_vector[32] = {0x00}, packed = 0;
   char output_file_name[MAX_FILE_NAME] = "", s[MAX_PATH] = "", temp[2*MAX_PATH] = "";
   struct node *tree_head = NULL;
   struct code code_values[MAX_CHARS] = {{-1, {0}, 0}};
   char* texto;
   size_t texto_s;
   long f_size;

   // revisar que haya archivo de entrada
   if (argc != 2) {
      printf("Formato: ./huffman filename\n");
      exit(1);
   }

   // abrir archivo ed entrada
   if ((file_in = fopen(argv[1], "r")) == NULL) {
      printf("No se pudo abrir el archivo.\n");
      exit(1);
   }

   // sacar frecuencia de cada caracter
   while ((c = fgetc(file_in)) != EOF) {
      freq[c]++;
   }

   // cerrar archivo de entrada
   fclose(file_in);

   for (count = 0; count < MAX_CHARS; count++) {
      if (freq[count] > 0) {
         bit_vector[count / 8] |= (1 << (count % 8));
      }
   }


   if ((strlen(argv[1])+strlen(".huff")) >= MAX_FILE_NAME) {
      printf("Input file name too long.  Output file cannot be generated.\n");
      exit(1);
   }

   // crear archivo de salida
   strncpy(output_file_name, argv[1], MAX_FILE_NAME);
   strncat(output_file_name, ".huff", MAX_FILE_NAME);

   // abrir archivo de salida
   if ((file_out = fopen(output_file_name, "w")) == NULL) {
      printf("Output file failed to open.\n");
      exit(1);
   }


   add_magic_num(file_out);
   add_bit_vector(file_out, bit_vector);
   num_bytes = add_size(file_out, freq);
   add_character_counts(file_out, freq, num_bytes);

   // construir arbol
   tree_head = generate_tree(freq);

   // generar huffman codes
   build_codes(tree_head, code_values, s, 0);

   // abrir arhchivo de entrada
   if ((file_in = fopen(argv[1], "r")) == NULL) {
      printf("Failed to open the input file.\n");
      exit(1);
   }


   //para tomar tiempo
   
   cudaEvent_t cpuI, cpuF;
   float cpuT;
   
   cudaEventCreate( &cpuI );
    cudaEventCreate( &cpuF );
    cudaEventRecord( cpuI, 0 );
     
   //para sacar tamano del archivo
   fseek(file_in, 0, SEEK_END);
   f_size = ftell(file_in);
   fseek(file_in, 0, SEEK_SET);
   
   //string con el archivo cargado en memoria
   char *string = (char *)malloc(sizeof(char) * f_size); 
   //arreglo con la longitud de la represnetacion en bits del caracter, sacado de la tabla de huffman
   int * offset = (int *)malloc(sizeof(int) * f_size);

   int i = 0;
   
   //llenar los arreglos
   while ((c = fgetc(file_in)) != EOF) {
      offset[i] = code_values[c].len;
      string[i] = c;
      i++;
      }
   char *d_string;
   
   //reservar memoria en tarjeta
   cudaMalloc<char>(&d_string, sizeof(char) * f_size);
   cudaMemcpy(d_string, string, f_size*sizeof(char), cudaMemcpyHostToDevice );
   
   
   thrust::device_vector<int> d_offset(offset, offset+f_size);   
  	
   //prefix sum en paralelo
   thrust::exclusive_scan(d_offset.begin(), d_offset.end(), d_offset.begin()); // in-place scan
    
   //copiar a host
   thrust::copy(d_offset.begin(), d_offset.end(), offset);   
   int last = offset[f_size - 1] + code_values[string[f_size-1]].len;
   //padding
   if (last%8 != 0)
   {
      last += 8-(last%8);
   }

   //memoria en device
   char * d_encoded_byte_stream;
   char * encoded_byte_stream = (char *)malloc(sizeof(char) * last);
   cudaMalloc<char>(&d_encoded_byte_stream, sizeof(char)*last);
   cudaMemset(d_encoded_byte_stream, '0', sizeof(char)*last);
   code * d_code_values;
   cudaMalloc<code>(&d_code_values, sizeof(code)*MAX_CHARS);   
   cudaMemcpy(d_code_values, code_values, MAX_CHARS*sizeof(code), cudaMemcpyHostToDevice );   
   
   //sacar pointer de thrust::device_vector
   int * d_offset2 = thrust::raw_pointer_cast( &d_offset[0] );
   //thrust::device_delete(d_offset2);
   int blocks;
   blocks = ceil((float)f_size/THREADS_PER_BLOCK);

   printf("blocks: %d\n", blocks);
   encode_byte_stream<<<blocks,THREADS_PER_BLOCK>>>(d_string, d_code_values, d_encoded_byte_stream, d_offset2, f_size);
   cudaFree(d_string);
   cudaFree(d_code_values);
   //cudaFree(d_offset2);
   
   //copiar a host
   cudaMemcpy(encoded_byte_stream, d_encoded_byte_stream, last*sizeof(char), cudaMemcpyDeviceToHost);
   
   int finalSize = last/8;
  
   //memoria en device
   unsigned char * d_encoded_bit_stream;
   //memoria en host
   unsigned char * encoded_bit_stream = (unsigned char *)malloc(finalSize * sizeof(unsigned char));
   
   cudaMalloc<unsigned char>(&d_encoded_bit_stream, sizeof(unsigned char)*finalSize);
   blocks = ceil((float)finalSize/THREADS_PER_BLOCK); //redondea al siguiente multiplo de 128
   compressed_bit_stream<<<blocks, THREADS_PER_BLOCK>>>(d_encoded_byte_stream, d_encoded_bit_stream, finalSize);
   cudaMemcpy(encoded_bit_stream, d_encoded_bit_stream, finalSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
   
   //escribe al archivo
   fwrite(encoded_bit_stream, sizeof(unsigned char), finalSize, file_out);
   
   //parar tiempo
   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF);
    
   printf("Tiempo %f: ", cpuT);

   //FREES
   cudaFree(d_encoded_byte_stream);
   cudaFree(d_encoded_bit_stream);
   
   free(encoded_bit_stream);
   free(encoded_byte_stream);
   free(string);
   free(offset);

   // cerrar archivo entrada
   if ((ret = fclose(file_in)) != 0) {
      printf("Failed to close the input file.");
   }

   //cerrar archivo salida
   if ((ret = fclose(file_out)) != 0) {
      printf("Failed to close the output file.");
   }

   free_tree(tree_head);

   return 0;
}

void add_magic_num(FILE *file) {

   unsigned char magic_num[4] = {0x4C,0x70,0xF0,0x7C}; //numero magico
   int i = 0, ret = 0;

   for (i = 0; i < 4; i++) {
      if ((ret = fprintf(file, "%c", magic_num[i])) != 1) {
         printf("Failure to add magic number to output file.\n"); //no se pudo agregar al archivo
         exit(1);
      }
   }

   return;
}

void add_bit_vector(FILE *file, unsigned char bit_vector[32]) {
   unsigned char c = 0;
   int i = 0, ret = 0;


   for (i = 0; i < 32; i++) {

      c = 0x00;
      c |= ((bit_vector[i] & 0x01) << 7);
      c |= ((bit_vector[i] & 0x02) << 5);
      c |= ((bit_vector[i] & 0x04) << 3);
      c |= ((bit_vector[i] & 0x08) << 1);
      c |= ((bit_vector[i] & 0x10) >> 1);
      c |= ((bit_vector[i] & 0x20) >> 3);
      c |= ((bit_vector[i] & 0x40) >> 5);
      c |= ((bit_vector[i] & 0x80) >> 7);

      if ((ret = fprintf(file,"%c", c)) != 1) {
         printf("Failure to output bit vector number.\n");
         exit(1);
      }
   }

   return;
}

int add_size(FILE *file, int freq[MAX_CHARS]) {

   int i = 0, num_bytes = 0, ret = 0;


   for (i = 0; i < MAX_CHARS; i++) {
      if (freq[i] & 0xFF000000) {
         num_bytes = 4;
      } else if ((freq[i] & 0x00FF0000) && num_bytes < 4) {
         num_bytes = 3;
      } else if ((freq[i] & 0x0000FF00) && num_bytes < 3) {
         num_bytes = 2;
      } else if ((freq[i] & 0x000000FF) && num_bytes < 2) {
         num_bytes = 1;
      }
   }


   if ((ret = fprintf(file, "%c", num_bytes)) != 1) {
      printf("Failure to output size of the frequency byte number.\n");
      exit(1);
   }

   return num_bytes;
}

void add_character_counts(FILE *file, int freq[MAX_CHARS], int num_bytes) {

   char *ptr = 0;
   int i = 0, ret = 0, j = 0;


   for (i = 0; i < MAX_CHARS; i++) {

      if (freq[i] == 0) {
         continue;
      }

      ptr = (char *)&freq[i];


      for (j = (num_bytes - 1); j >= 0 ; j--) {

         if ((ret = fprintf(file, "%c", ptr[j])) != 1) {
            printf("Failure to output the freqency byte.\n");
            exit(1);
         }
      }
   }

   return;
}

