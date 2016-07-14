#include <iostream>

#include <cuda.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

__global__
void blur_shared(const double *in, double* out, int n) {
    extern __shared__ double buffer[];

    auto block_start = blockDim.x * blockIdx.x;
    auto li = threadIdx.x + 1;
    auto gi = li + block_start;

    if(gi<n-1) {
        // load shared memory
        buffer[li] = in[gi];
        if(li==1) {
            buffer[0] = in[block_start];
            buffer[blockDim.x+1] = in[block_start+blockDim.x+1];
        }

        __syncthreads();

        out[gi] = 0.25*(buffer[li-1] + 2.0*buffer[li] + buffer[li+1]);
    }
}

__global__
void blur_shared_block(const double *in, double* out, int n) {
    extern __shared__ double buffer[];

    auto i = threadIdx.x + 1;

    if(i<n-1) {
        // load shared memory
        buffer[i] = in[i];
        if(i==1) {
            buffer[0] = in[0];
            buffer[n] = in[n];
        }

        __syncthreads();

        out[i] = 0.25*(buffer[i-1] + 2.0*buffer[i] + buffer[i+1]);
    }
}

__global__
void blur(const double *in, double* out, int n) {
    auto i = threadIdx.x + blockDim.x * blockIdx.x + 1;

    if(i<n-1) {
        out[i] = 0.25*(in[i-1] + 2.0*in[i] + in[i+1]);
    }
}

int main(int argc, char** argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    size_t n = 1 << pow;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "blur 1D test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    auto x_host = malloc_host<double>(n+2, 0.);
    // set boundary conditions to 1
    x_host[0]   = 1.0;
    x_host[n+1] = 1.0;

    auto x0 = malloc_device<double>(n+2);
    auto x1 = malloc_device<double>(n+2);

    // copy initial conditions to device
    copy_to_device<double>(x_host, x0, n+2);
    copy_to_device<double>(x_host, x1, n+2);

    // find the launch grid configuration
    auto block_dim = 512ul;
    auto grid_dim = (n+(block_dim-1))/block_dim;

    blur<<<grid_dim, block_dim>>>(x0, x1, n);

    CudaStream stream;
    auto start_event = stream.enqueue_event();
    for(auto step=0; step<nsteps; ++step) {
        //blur<<<grid_dim, block_dim>>>(x0, x1, n);
        blur_shared<<<grid_dim, block_dim, (block_dim+2)*sizeof(double)>>>(x0, x1, n);
        std::swap(x0, x1);
    }
    auto stop_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host<double>(x0, x_host, n+2);

    stop_event.wait();
    auto time = stop_event.time_since(start_event);
    std::cout << "==== that took " << time << " seconds"
              << " (" << time/nsteps << "s/step)" << std::endl;


    for(auto i=0; i<20; ++i) std::cout << x_host[i] << " "; std::cout << std::endl;

    return 0;
}

