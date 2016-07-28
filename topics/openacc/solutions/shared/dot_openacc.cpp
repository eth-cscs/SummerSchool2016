#include <iostream>

#define NO_CUDA
#include "util.h"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

double dot_gpu(const double *x, const double* y, int n) {
    double sum = 0;
    int i;

#pragma acc parallel loop copyin(x[0:n]) copyin(y[0:n]) reduction(+:sum)
    for(i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }

    return sum;
}

int main(int argc, char** argv) {
    size_t n  = read_arg(argc, argv, 1, 4);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product OpenACC of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto result   = dot_gpu(x_h, y_h, n);
    auto expected = dot_host(x_h, y_h, n);
    std::cout << "expected " << expected << " got " << result << "\n";
    return 0;
}
