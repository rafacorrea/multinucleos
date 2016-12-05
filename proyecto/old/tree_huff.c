
#include <stdlib.h> 
#include <string.h>  

#include "tree_huff.h"

struct node *generate_tree(int freq[MAX_CHARS]) {

   int count = 0;
   struct node *head = NULL;


   for (count = 0; count < MAX_CHARS; count++) {
      if (freq[count] != 0) {
         head = insert_ordered(head, count, freq[count], NULL, NULL, 0);
      }
   }

   if (head == NULL) {
      return NULL;
   }


   while (head->p_next != NULL) {
      head = insert_ordered(head, -1, head->freq + (head->p_next)->freq, head, head->p_next, 1);
      head = (head->p_next)->p_next;
   }

   return head;
}

struct node *insert_ordered(struct node *old_head, int ch, int freq,
      struct node *left, struct node *right, int tree) {


   struct node *new_head;


   if (old_head == NULL) {
      // crear neuvo nodo
      struct node *insert_node = (struct node *)malloc(sizeof(struct node));

      insert_node->ch = ch;
      insert_node->freq = freq;
      insert_node->p_left = left;
      insert_node->p_right = right;
      insert_node->p_next = NULL;
      new_head = insert_node;
   }
   // insertar nodo
   else if ((tree == 0) && (freq < old_head->freq)) {
      struct node *insert_node = (struct node *)malloc(sizeof(struct node));
      insert_node->ch = ch;
      insert_node->freq = freq;
      insert_node->p_left = left;
      insert_node->p_right = right;
      insert_node->p_next = old_head;
      new_head = insert_node;
   }
   else if ((tree == 1) && (freq <= old_head->freq)) {

      struct node *insert_node = (struct node *)malloc(sizeof(struct node));

      insert_node->ch = ch;
      insert_node->freq = freq;
      insert_node->p_left = left;
      insert_node->p_right = right;
      insert_node->p_next = old_head;
      new_head = insert_node;
   }

   else {
      new_head = old_head;
      new_head->p_next = insert_ordered(old_head->p_next, ch, freq, left, right, tree);
   }

   return new_head;
}

void build_codes(struct node *tree_head, struct code code_values[MAX_CHARS],
      char code[MAX_PATH], int code_len) {


   if (tree_head == NULL) {
      return;
   }


   if ((tree_head->p_right == NULL) && (tree_head->p_left == NULL)) {
      code_values[tree_head->ch].ch = tree_head->ch;
      strncpy(code_values[tree_head->ch].path, code, MAX_PATH);
      code_values[tree_head->ch].len = code_len;
   }

   else {
      char tmp[MAX_CHARS] = "";
      strncpy(tmp, code, MAX_CHARS);
      code_len++;
      build_codes(tree_head->p_left, code_values, strncat(code, "0", 1), code_len);
      build_codes(tree_head->p_right, code_values, strncat(tmp, "1", 1), code_len);
   }

   return;
}

void free_tree(struct node *tree_head) {

   if (tree_head != NULL) {
      free_tree(tree_head->p_left);
      free_tree(tree_head->p_right);
      free(tree_head);
   }

   return;
}

