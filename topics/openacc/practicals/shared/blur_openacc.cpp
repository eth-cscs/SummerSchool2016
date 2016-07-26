#include <iostream>
#include <cassert>

#include <omp.h>

#define NO_CUDA
#include "util.h"

// TODO: This must be called from an OpenACC region
double blur(int pos, const double *u)
{
    return 0.25*(u[pos-1] + 2.0*u[pos] + u[pos+1]);
}

void blur_twice_gpu_naive(double* in , double* out , int n, int nsteps)
{
    double* buffer = malloc_host<double>(n);

    for (auto istep = 0; istep < nsteps; ++istep) {
        int i;

        // TODO: Offload the following loop to GPU
        for(i=1; i<n-1; ++i) {
            buffer[i] = blur(i, in);
        }

        // TODO: Offload the following loop to GPU
        for(i=2; i<n-2; ++i) {
            out[i] = blur(i, buffer);
        }

        // swap in/out
        std::swap(in, out);
    }

    free(buffer);
}

void blur_twice_gpu_nocopies(double *in , double *out , int n, int nsteps)
{
    double *buffer = malloc_host<double>(n);

    // TODO: Move all data to GPU before starting the computation
    for (auto istep = 0; istep < nsteps; ++istep) {
        int i;

        // TODO: Offload this loop to GPU
        for(i = 1; i < n-1; ++i) {
            buffer[i] = blur(i, in);
        }

        // TODO: Offload this loop to GPU
        for(i = 2; i < n-2; ++i) {
            out[i] = blur(i, buffer);
        }

        // TODO: Copy in to out; pointer swap is not supported
    }

    free(buffer);
}

int main(int argc, char **argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    size_t n = (1 << pow) + 4;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dispersion 1D test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    auto x0 = malloc_host<double>(n+4, 0.);
    auto x1 = malloc_host<double>(n+4, 0.);

    // set boundary conditions to 1
    x0[0]   = x1[0]   = 1.0;
    x0[1]   = x1[1]   = 1.0;
    x0[n-2] = x1[n-2] = 1.0;
    x0[n-1] = x1[n-1] = 1.0;

    auto tstart = get_time();
    // TODO: Make sure to call the correct function
    blur_twice_gpu_naive(x0, x1, n, nsteps);
    auto time = get_time() - tstart;

    for(auto i=0; i<std::min(decltype(n){20},n); ++i) {
        std::cout << x1[i] << " ";
    }

    std::cout << "\n";
    std::cout << "==== that took " << time << " seconds"
              << " (" << time/nsteps << "s/step)" << std::endl;

    return 0;
}
