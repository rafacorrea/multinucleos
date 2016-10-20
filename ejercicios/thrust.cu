#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <chrono>
#include <algorithm>
#include <vector>

//using sys_clock = std::chrono::system_clock;

int thrust_sequence()
{
    thrust::device_vector<int> D_vec(10,1);
    thrust::fill(D_vec.begin(), D_vec.begin()+7, 9);
    thrust::host_vector<int> H_vec(D_vec.begin(),D_vec.begin()+5);
    thrust::sequence(H_vec.begin(), H_vec.end(), 5, 2);
    thrust::copy(H_vec.begin(), H_vec.end(), D_vec.begin());
    
    int i = 0;
    for(auto value : D_vec)
        std::cout << "D[" << i++ << "]= " << value << std::endl;
}

int thrust_sort()
{
    int current_h = 0, current_d = 0, exit = 0, limit = 1 << 24;
    
    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli> exec_time_ms;
    
    thrust::host_vector<int> h_vec(limit);
    
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    
    thrust::device_vector<int> d_vec = h_vec;
    
    t1 = std::chrono::system_clock::now();
    thrust::sort(d_vec.begin(), d_vec.end());
    t2 = std::chrono::system_clock::now();
    
    exec_time_ms = t2 - t1;
    
    std::cout << "thrust gpu sort: " << exec_time_ms.count() << "ms." << std::endl;
    
    std::vector<int> stl_vec(h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), stl_vec.begin());
    
    t1 = std::chrono::system_clock::now();
    std::sort(stl_vec.begin(), stl_vec.end());
    t2 = std::chrono::system_clock::now();
    
    exec_time_ms = t2 - t1;
    
    std::cout << "stl sort: " << exec_time_ms.count() << "ms." << std::endl;
}


struct functor
{
    const float a;
    functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(const float &x, const float &y) const
    {
        return a * x + y;
    }

};

int operador()
{
    const float A = 5;
    const int size = 10;
    
    thrust::host_vector<float> X(size), Y(size), Z(size);
    
    thrust::sequence(X.begin(), X.end(), 10, 10);
    thrust::sequence(Y.begin(), Y.end(), 1, 5);
    
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), functor(A));
    
    for(int i = 0; i < Y.size(); i++)
    {
        std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
    }
}


template <typename T>
struct square
{
    __host__ __device__ float operator()(const T &x) const
    {
        return x*x;
    }

};


int main ()
{
    float x[4] = {1.0, 2.0, 3.0, 4.0};
    thrust::device_vector<float> d_vec(x, x+4);
    square<float> unary_op;
    thrust::plus<float> binary_op;
    
    float norm = std::sqrt(
        thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0, binary_op)
    );
    
    std::cout << norm << std::endl;
}

