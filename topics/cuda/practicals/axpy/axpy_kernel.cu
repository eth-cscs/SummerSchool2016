#include <iostream>

#include <cuda.h>
//#include <cuda_runtime.h>

#include "util.h"

// TODO CUDA kernel implementing axpy
//      y = y + alpha*x
//void axpy(int n, double alpha, const double* x, double* y)

template <typename F>
int calculate_threads_per_block(F kernel, int  n) {
    int block_dim, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_dim, kernel, 0, n);
    std::cout << "++++ suggested block_dim " << block_dim
              << " and " << min_grid_size << " blocks"
              << std::endl;
    return block_dim;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    auto size_in_bytes = n * sizeof(double);

    cuInit(0);

    std::cout << "memcopy and daxpy test of size " << n << std::endl;

    double* x_device = malloc_device<double>(n);
    double* y_device = malloc_device<double>(n);

    double* x_host = malloc_host<double>(n, 1.5);
    double* y_host = malloc_host<double>(n, 3.0);
    double* y      = malloc_host<double>(n, 0.0);

    // copy to device
    auto start = get_time();
    copy_to_device<double>(x_host, x_device, n);
    copy_to_device<double>(y_host, y_device, n);
    auto time_H2D = get_time() - start;

    // TODO calculate grid dimensions
    // IGNORE for the first kernel writing exercise
    // don't try to use cudaOccupancyMaxPotentialBlockSize the first time

    // the cudaThreadSynchronize() functions synchronize the host and device
    // so that the timings are accurate
    cudaThreadSynchronize();

    start = get_time();
    // TODO launch kernel (alpha=2.0)

    cudaThreadSynchronize();
    auto time_axpy = get_time() - start;

    // check for error in last kernel call
    cuda_check_last_kernel("axpy kernel");

    // copy result back to host
    start = get_time();
    copy_to_host<double>(y_device, y, n);
    auto time_D2H = get_time() - start;

    std::cout << "-------\ntimings\n-------" << std::endl;
    std::cout << "H2D  : " << time_H2D << " s" << std::endl;
    std::cout << "D2H  : " << time_D2H << " s" << std::endl;
    std::cout << "axpy : " << time_axpy << " s" << std::endl;
    std::cout << std::endl;
    std::cout << "total: " << time_axpy+time_H2D+time_D2H << " s" << std::endl;
    std::cout << std::endl;

    std::cout << "-------\nbandwidth\n-------" << std::endl;
    auto H2D_BW = size_in_bytes * 2.0 / time_H2D / (1024*1024);
    auto D2H_BW = size_in_bytes / time_D2H / (1024*1024);
    std::cout << "H2D BW : " << H2D_BW << " MB/s" << std::endl;
    std::cout << "D2H BW : " << D2H_BW << " MB/s" << std::endl;

    // check for errors
    auto errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for(auto i=0; i<n; ++i) {
        if(std::fabs(6.-y[i])>1e-15) {
            errors++;
        }
    }

    if(errors>0) {
        std::cout << "\n============ FAILED with " << errors << " errors" << std::endl;
    }
    else {
        std::cout << "\n============ PASSED" << std::endl;
    }

    cudaFree(x_device);
    cudaFree(y_device);

    free(x_host);
    free(y_host);
    free(y);

    return 0;
}

