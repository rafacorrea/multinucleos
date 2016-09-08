#include <stdio.h>
#include <stdlib.h>



void main()
{
	 int n =0;
	printf("Dame un núm: ");
	scanf ("%d", &n);

	int i=0;
	for (i=10; i<=1024; i++)
	{
		if (n % i == 0)
			{
			  printf("%d  -----  %d\n", i , n/i);
			}

	}




}
