#include <stdio.h>   
#include <string.h>  
#include <stdlib.h>  

#include "tree_huff.cuh"

#define FILE_NAME_MAX_LEN 270
#define MAX_FILE_NAME 256


void add_magic_num(FILE *file);
void add_bit_vector(FILE *file, unsigned char bit_vector[32]);
int  add_size(FILE *file, int freq[MAX_CHARS]);
void add_character_counts(FILE *file, int freq[MAX_CHARS], int num_bytes);

int main(int argc, char *argv[]) {


   FILE *file_in, *file_out;
   int freq[MAX_CHARS] = {0}, c = 0, count = 0, num_bytes = 0, ret = 0;
   unsigned char bit_vector[32] = {0x00}, packed = 0;
   char output_file_name[MAX_FILE_NAME] = "", s[MAX_PATH] = "", temp[2*MAX_PATH] = "";
   struct node *tree_head = NULL;
   struct code code_values[MAX_CHARS] = {{-1, {0}, 0}};


   // revisar que haya archivo de entrada
   if (argc != 2) {
      printf("Formato: ./huffman filename\n");
      exit(1);
   }

   // abrir archivo ed entrada
   if ((file_in = fopen(argv[1], "r")) == NULL) {
      printf("No se udo abrir el archivo.\n");
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


   cudaEvent_t cpuI, cpuF;
   float cpuT;
   
   cudaEventCreate( &cpuI );
    cudaEventCreate( &cpuF );
    cudaEventRecord( cpuI, 0 );
   // basado en el codigo huffman llenar archivo
   while ((c = fgetc(file_in)) != EOF) {
      strncat(temp, code_values[c].path, 2*MAX_PATH);


      if (strlen(temp) < 8) {
         continue;
      }

      // escribir a la salida
      while (strlen(temp) > 8) {
         for (count = 0; count < 8; count++) {
            if (temp[count] == '0') {
               packed <<= 1;
            } else {
               packed = (packed << 1) | 0x01;
            }
         }

         fwrite(&packed, sizeof(unsigned char), 1, file_out);
         packed = 0;
         strcpy(temp, &temp[8]);
      }
   }

   packed = 0;


   if (strlen(temp) > 0) {
      
      for (count = 0; count < strlen(temp); count++) {
         if (temp[count] == '0') {
            packed <<= 1;
         } else {
            packed = (packed << 1) | 0x01;
         }
      }

      // padding
      for (count = strlen(temp); count < 8; count++) {
         packed = (packed << 1) ;
      }

      // escribir al archivo
      fwrite(&packed, sizeof(unsigned char), 1, file_out);
   }
   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF);
    
   printf("Tiempo %f: ", cpuT);
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

   unsigned char magic_num[4] = {0x4C,0x70,0xF0,0x7C};
   int i = 0, ret = 0;

   for (i = 0; i < 4; i++) {
      if ((ret = fprintf(file, "%c", magic_num[i])) != 1) {
         printf("Failure to add magic number to output file.\n");
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

