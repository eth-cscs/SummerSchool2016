#include <chrono>
#include <iostream>

#define NO_CUDA
#include "util.h"

// OpenMP implementation of axpy kernel
void axpy(int n, double alpha, const double *x, double* y) {
    #pragma omp parallel for
    #pragma ivdep
    for(auto i=0; i<n; ++i) {
        y[i] += alpha*x[i];
    }
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;

    std::cout << "memcopy and daxpy test of size " << n << std::endl;

    double* x = malloc_host<double>(n, 1.5);
    double* y = malloc_host<double>(n, 3.0);

    // use dummy fields to avoid cache effects, which make results harder to interpret
    // use 1<<24 to ensure that cache is completely purged for all n
    double* x_ = malloc_host<double>(1<<24, 1.5);
    double* y_ = malloc_host<double>(1<<24, 3.0);
    axpy(1<<24, 2.0, x_, y_);

    auto start = get_time();
    axpy(n, 2.0, x, y);
    auto time_axpy = get_time() - start;

    auto BW = 3 * n * sizeof(double) / 1e9 / time_axpy;
    std::cout << "-------\ntimings\n-------" << std::endl;
    std::cout << "::axpy : " << time_axpy << " s" << std::endl;
    std::cout << "::BW   : " << BW << " GB/s" << std::endl;
    std::cout << std::endl;

    // check for errors
    auto errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }

    if(errors>0) std::cout << "\n============ FAILED with " << errors << " errors" << std::endl;
    else         std::cout << "\n============ PASSED" << std::endl;

    free(x);
    free(y);

    return 0;
}


