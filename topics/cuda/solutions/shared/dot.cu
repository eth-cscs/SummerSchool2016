#include <iostream>

#include <cuda.h>

#include "util.h"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// TODO implement dot product kernel
// hint : the result should be a single value in result[0]
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n) {
    extern __shared__ double buffer[];

    auto i = threadIdx.x;
    if(i<n) {
        // each thread calculates and stores contribution to the sum
        buffer[i] = x[i]*y[i];

        // do binary reduction iteratively
        auto m = n>>1;
        while(m) {
            __syncthreads();
            if(i<m) {
                buffer[i] += buffer[i+m];
            }
            m>>=1;
        }

        // thread 0 writes the result
        if(i==0) {
            result[0]= buffer[0];
        }
    }
}

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_device<double>(1);
    dot_gpu_kernel<<<1, n, n*sizeof(double)>>>(x, y, result, n);
    double r;
    copy_to_host<double>(result, &r, 1);
    return r;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    std::cout << "expected " << expected << " got " << result << std::endl;

    return 0;
}

