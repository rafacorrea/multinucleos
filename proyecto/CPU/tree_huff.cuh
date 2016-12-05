
#ifndef HUFFMAN_TREE
#define HUFFMAN_TREE

#define MAX_CHARS 256
#define MAX_PATH  5000

struct node {
   int ch;
   int freq;
   struct node *p_left;
   struct node *p_right;
   struct node *p_next;
};

struct code {
   int ch;
   char path[MAX_PATH];
   int len;
};


struct node *generate_tree(int freq[MAX_CHARS]);
struct node *insert_ordered(struct node *old_head, int ch, int freq, struct node *left, struct node *right, int tree);
void build_codes(struct node *tree_head, struct code code_values[MAX_CHARS], char path[MAX_PATH], int code_len);
void free_tree(struct node *tree_head);

#endif //HUFFMAN_TREE
