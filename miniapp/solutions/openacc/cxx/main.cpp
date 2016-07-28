// ******************************************
// implicit time stepping implementation of 2D diffusion problem
// Ben Cumming, CSCS
// Vasileios Karakasis, CSCS
// *****************************************

// A small benchmark app that solves the 2D fisher equation using second-order
// finite differences.

// Syntax: ./main nx ny nt t

#include <algorithm>
#include <fstream>
#include <iostream>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <omp.h>
#include <stdio.h>

#include "data.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

using namespace data;
using namespace linalg;
using namespace operators;
using namespace stats;

// ==============================================================================

// set the initial condition
// a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
// no larger than 1/8 of both xdim and ydim
void initial_condition(Field &x_new, int nx, int ny)
{
    ss_fill(x_new, 0., nx*ny);

    double xc = 1.0 / 4.0;
    double yc = (ny - 1) * options.dx / 4;
    double radius = fmin(xc, yc) / 2.0;

    #pragma acc parallel present(x_new)
    {
        #pragma acc loop
        for (int j = 0; j < ny; j++) {
            double y = (j - 1) * options.dx;
            #pragma acc loop
            for (int i = 0; i < nx; i++) {
                double x = (i - 1) * options.dx;
                if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
                    x_new(i, j) = 0.1;
            }
        }
    }
}


// read command line arguments
static void readcmdline(Discretization& options, int argc, char* argv[])
{
    if (argc<5 || argc>6)
    {
        printf("Usage: main nx ny nt t verbose\n");
        printf("  nx        number of gridpoints in x-direction\n");
        printf("  ny        number of gridpoints in y-direction\n");
        printf("  nt        number of timesteps\n");
        printf("  t         total time\n");
        printf("  verbose   (optional) verbose output\n");
        exit(1);
    }

    // read nx
    options.nx = atoi(argv[1]);
    if (options.nx < 1)
    {
        fprintf(stderr, "nx must be positive integer\n");
        exit(-1);
    }

    // read ny
    options.ny = atoi(argv[2]);
    if (options.ny < 1)
    {
        fprintf(stderr, "ny must be positive integer\n");
        exit(-1);
    }

    // read nt
    options.nt = atoi(argv[3]);
    if (options.nt < 1)
    {
        fprintf(stderr, "nt must be positive integer\n");
        exit(-1);
    }

    // read total time
    double t = atof(argv[4]);
    if (t < 0)
    {
        fprintf(stderr, "t must be positive real value\n");
        exit(-1);
    }

    // set verbosity if requested
    verbose_output = false;
    if (argc==6) {
        verbose_output = true;
    }

    // store the parameters
    options.N = options.nx * options.ny;

    // compute timestep size
    options.dt = t / options.nt;

    // compute the distance between grid points
    // assume that x dimension has length 1.0
     options.dx = 1. / (options.nx - 1);

    // set alpha, assume diffusion coefficient D is 1
    options.alpha = (options.dx * options.dx) / (1. * options.dt);
}

// ==============================================================================

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(options, argc, argv);
    int nx = options.nx;
    int ny = options.ny;
    int N  = options.N;
    int nt = options.nt;

    // set iteration parameters
    int max_cg_iters     = 200;
    int max_newton_iters = 50;
    double tolerance     = 1.e-6;

    std::cout << "========================================================================" << std::endl;
    std::cout << "                      Welcome to mini-stencil!" << std::endl;
    std::cout << "version   :: OpenACC C++" << std::endl;
    std::cout << "mesh      :: " << options.nx << " * " << options.ny << " dx = " << options.dx << std::endl;
    std::cout << "time      :: " << nt << " time steps from 0 .. " << options.nt*options.dt << std::endl;;
    std::cout << "iteration :: " << "CG "          << max_cg_iters
                                 << ", Newton "    << max_newton_iters
                                 << ", tolerance " << tolerance << std::endl;;
    std::cout << "========================================================================" << std::endl;

    // allocate global fields
    // allocate global fields
    x_new.init(nx,ny);
    x_old.init(nx,ny);
    bndN.init(nx,1);
    bndS.init(nx,1);
    bndE.init(ny,1);
    bndW.init(ny,1);

    Field b(nx,ny);
    Field deltax(nx,ny);

    // set dirichlet boundary conditions to 0 all around
    ss_fill(bndN, 0., nx);
    ss_fill(bndS, 0., nx);
    ss_fill(bndE, 0., ny);
    ss_fill(bndW, 0., ny);
    ss_fill(b, 0., nx*ny);
    ss_fill(deltax, 0., nx*ny);

    initial_condition(x_new, nx, ny);

    iters_cg = 0;
    iters_newton = 0;

    // start timer
    #pragma acc wait
    double timespent = -omp_get_wtime();

    // main timeloop
    for (int timestep = 1; timestep <= nt; timestep++)
    {
        // set x_new and x_old to be the solution
        ss_copy(x_old, x_new, N);

        double residual;
        bool converged = false;
        int it;
        for (it=0; it<max_newton_iters; it++)
        {
            // compute residual : requires both x_new and x_old
            diffusion(x_new, b);

            residual = ss_norm2(b, N);

            // check for convergence
            if (residual < tolerance)
            {
                converged = true;
                break;
            }

            // solve linear system to get -deltax
            bool cg_converged = false;
            ss_cg(deltax, b, max_cg_iters, tolerance, cg_converged);

            // check that the CG solver converged
            if (!cg_converged) break;

            // update solution
            ss_axpy(x_new, -1.0, deltax, N);
        }
        iters_newton += it+1;

        // output some statistics
        if (converged && verbose_output) {
            std::cout << "step " << timestep
                      << " required " << it
                      << " iterations for residual " << residual << "\n";
        }
        if (!converged) {
            std::cerr << "step " << timestep
                      << " ERROR : nonlinear iterations failed to converge\n";
            break;
        }
    }

    // get times
    #pragma acc wait
    timespent += omp_get_wtime();

    // Update the host copy
    x_new.update_host();

    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

    // binary data
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(x_new.data(), sizeof(double), nx * ny, output);
        fclose(output);
    }

    std::ofstream fid("output.bov");
    fid << "TIME: 0.0\n";
    fid << "DATA_FILE: output.bin\n";
    fid << "DATA_SIZE: " << options.nx << ", " << options.ny << ", 1\n";
    fid << "DATA_FORMAT: DOUBLE\n";
    fid << "VARIABLE: phi\n";
    fid << "DATA_ENDIAN: LITTLE\n";
    fid << "CENTERING: nodal\n";
    fid << "BRICK_SIZE: 1.0 " << (options.ny-1)*options.dx << " 1.0\n";

    // print table sumarizing results
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "simulation took " << timespent << " seconds\n";
    std::cout << int(iters_cg) << " conjugate gradient iterations, at rate of "
              << float(iters_cg)/timespent << " iters/second\n";
    std::cout << iters_newton << " newton iterations\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::cout << "Goodbye!\n";
    return 0;
}
