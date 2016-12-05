#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define SIZE 1050

int main()
{
    int * data = (int *)malloc(SIZE * sizeof(int));// = {1,2,3,4,5,6,7,8,9};
    for (int i = 0; i < SIZE; i++)
        data[i] = i;

    
  	thrust::device_vector<int> d_data(data, data+SIZE);   
  	
    thrust::exclusive_scan(d_data.begin(), d_data.end(), d_data.begin()); // in-place scan
    
    thrust::copy(d_data.begin(), d_data.end(), data);    
    
    for(int i = 0; i < SIZE; i++)
    {
        printf("%d\n",data[i]);
    }
}
