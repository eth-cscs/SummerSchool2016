#include <iostream>

#include <cuda.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

__global__
void blur_twice(const double *in, double* out, int n) {
    extern __shared__ double buffer[];

    auto block_start = blockDim.x * blockIdx.x;
    auto block_end   = block_start + blockDim.x;

    auto lid = threadIdx.x;
    auto gid = lid + block_start;

    auto blur = [] (int pos, double const* field) {
        return 0.25*(field[pos-1] + 2.0*field[pos] + field[pos+1]);
    };

    if(gid<n-4) {
        auto li = lid+2;
        auto gi = gid+2;

        buffer[li] = blur(gi, in);
        if(threadIdx.x==0) {
            buffer[1] = blur(block_start+1, in);
            buffer[blockDim.x+2] = blur(block_end+2, in);
        }

        __syncthreads();

        out[gi] = blur(li, buffer);
    }
}

int main(int argc, char** argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    size_t n = (1 << pow) + 4;

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dispersion 1D test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    auto x_host = malloc_host<double>(n, 0.);
    // set boundary conditions to 1
    x_host[0]   = 1.0;
    x_host[1]   = 1.0;
    x_host[n-2] = 1.0;
    x_host[n-1] = 1.0;

    auto x0 = malloc_device<double>(n);
    auto x1 = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_host, x0, n);
    copy_to_device<double>(x_host, x1, n);

    // find the launch grid configuration
    auto block_dim = 256;
    auto grid_dim = (n-4)/block_dim + ((n-4)%block_dim ? 1 : 0);
    auto shared_size = sizeof(double)*(block_dim+4);

    std::cout << "threads per block " << block_dim
              << ", in " << grid_dim << " blocks"
              << std::endl;

    CudaStream stream;
    auto start_event = stream.enqueue_event();
    for(auto step=0; step<nsteps; ++step) {
        blur_twice<<<grid_dim, block_dim, shared_size>>>(x0, x1, n);
        std::swap(x0, x1);
    }
    auto stop_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host<double>(x0, x_host, n);

    for(auto i=0; i<std::min(decltype(n){20},n); ++i) std::cout << x_host[i] << " "; std::cout << std::endl;

    stop_event.wait();
    auto time = stop_event.time_since(start_event);
    std::cout << "==== that took " << time << " seconds"
              << " (" << time/nsteps << "s/step)" << std::endl;

    return 0;
}

