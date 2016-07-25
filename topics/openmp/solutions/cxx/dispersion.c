#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>

#define INDEX(I, J, IDIM) ((I)+(J)*(IDIM))

int scanArgs(int argc, char** argv, int *nx, int *ny, int *nt, int *blocky);

/////////////////////////////////////////////////
// print the solution to the screen
/////////////////////////////////////////////////
void printSolution(double *s, int iDim, int jDim){
    for(int j=0; j<jDim; j++){
        for(int i=0; i<iDim; i++){
            printf("%8.6f ", s[j*iDim + i]);
        }
        printf("\n");
    }
    printf("\n");
}

/////////////////////////////////////////////////
// set initial conditions
// boundary condition of 1
// interior values of 0
/////////////////////////////////////////////////
void setInitialConditions(double *s, int iDim, int jDim){
    int i, j;

    // intialise values at the interior
    for(int j=0; j<jDim; j++){
        int k=j*iDim;
        for(int i=0; i<iDim; i++, k++){
            s[k] = 0.;
        }
    }

    ///////////////////////////////////////////////
    // set the boundary values in the halo to 1
    // the halo is 2 grid points wide
    ///////////////////////////////////////////////
    // top and bottom boundaries
    for(i=0; i<iDim; i++){
        s[i] = 1.;
        s[i+iDim] = 1.;
        s[i+(jDim-1)*iDim]  = 1.;
        s[i+(jDim-2)*iDim]  = 1.;
    }
    // left and right boundaries
    for(j=2; j<jDim-2; j++){
        int k=j*iDim;
        // set left boundary to 1
        s[k] = 1.;
        s[k+1] = 1.;
        // set right boundary to 1
        s[k+iDim-1] = 1.;
        s[k+iDim-2] = 1.;
    }
}

//////////////////////////////////////////////////////////
// version 1 of dispersion stencil
//
// sOut = sIn + D \Nabla^4 SIn
//////////////////////////////////////////////////////////
void dispersionV1(double *sIn, double *sOut, double *buffer,  double D, int iDim, int jDim){
    // compute the laplacian of the input field, and store in buffer field
    #pragma omp parallel for
    for(int j=1; j<jDim-1; j++){
        double *sP   = sIn    + INDEX(0,j,iDim);
        double *sN   = sIn    + INDEX(0,j+1,iDim);
        double *sS   = sIn    + INDEX(0,j-1,iDim);
        double *b    = buffer + INDEX(0,j,iDim);
        #pragma ivdep
        for(int i=1; i<iDim-1; i++){
            b[i] =  -4.*sP[i] + sP[i-1] + sP[i+1] + sN[i] + sS[i];
        }
    }
    // compute the disspersion result using
    #pragma omp parallel for
    for(int j=2; j<jDim-2; j++){
        double *bP   = buffer + INDEX(0,j,iDim);
        double *bN   = buffer + INDEX(0,j+1,iDim);
        double *bS   = buffer + INDEX(0,j-1,iDim);
        double *s    = sOut   + INDEX(0,j,iDim);
        double *sP   = sIn    + INDEX(0,j,iDim);
        #pragma ivdep
        for(int i=2; i<iDim-2; i++){
            s[i] =  sP[i] - D*(-4.*bP[i] + bP[i-1] + bP[i+1] + bN[i] + bS[i]);
        }
    }
}

//////////////////////////////////////////////////////////
// version 2 of dispersion stencil
//
// sOut = sIn + D \Nabla^4 SIn
//////////////////////////////////////////////////////////
void dispersionV2(double *sIn, double *sOut, double *buffer,  double D, int iDim, int jDim, int blockDim){
    int numBlocks = (jDim-4)/blockDim;
    #pragma omp parallel for
    for(int block=0; block<numBlocks; block++){
        int jStart = block*blockDim + 2;
        int jEnd   = jStart + blockDim;;

        // compute the laplacian of the input field on the block
        // store result in buffer
        for(int j=jStart-1; j<jEnd+1; j++){
            double *sP   = sIn    + INDEX(0,j,iDim);
            double *sN   = sIn    + INDEX(0,j+1,iDim);
            double *sS   = sIn    + INDEX(0,j-1,iDim);
            double *b    = buffer + INDEX(0,j,iDim);
            #pragma ivdep
            for(int i=1; i<iDim-1; i++){
                b[i] =  -4.*sP[i] + sP[i-1] + sP[i+1] + sN[i] + sS[i];
            }
        }
        for(int j=jStart; j<jEnd; j++){
            double *bP   = buffer + INDEX(0,j,iDim);
            double *bN   = buffer + INDEX(0,j+1,iDim);
            double *bS   = buffer + INDEX(0,j-1,iDim);
            double *s    = sOut   + INDEX(0,j,iDim);
            double *sP   = sIn    + INDEX(0,j,iDim);
            #pragma ivdep
            for(int i=2; i<iDim-2; i++){
                s[i] =  sP[i] - D*(-4.*bP[i] + bP[i-1] + bP[i+1] + bN[i] + bS[i]);
            }
        }
    }
}

//////////////////////////////////////////////////////////
// version 3 of dispersion stencil
//
// sOut = sIn + D \Nabla^4 SIn
//////////////////////////////////////////////////////////
void dispersionV3(double *sIn, double *sOut, double **buffers,  double D, int iDim, int jDim, int blockDim){
    int numBlocks = (jDim-4)/blockDim;
    #pragma omp parallel
    {
        // get a pointer to the thread-private buffer
        int threadID = omp_get_thread_num();
        double *buffer = buffers[threadID];
        #pragma omp for
        for(int block=0; block<numBlocks; block++){
            int jStart = block*blockDim + 2;
            int jEnd   = jStart + blockDim;;

            // compute the laplacian of the input field on the block
            // store result in buffer
            int jb = 0;
            for(int j=jStart-1; j<jEnd+1; j++, jb++){
                double *sP   = sIn    + INDEX(0,j,iDim);
                double *sN   = sIn    + INDEX(0,j+1,iDim);
                double *sS   = sIn    + INDEX(0,j-1,iDim);
                double *b    = buffer + INDEX(0,jb,iDim);
                #pragma ivdep
                for(int i=1; i<iDim-1; i++){
                    b[i] =  -4.*sP[i] + sP[i-1] + sP[i+1] + sN[i] + sS[i];
                }
            }
            jb = 1;
            for(int j=jStart; j<jEnd; j++, jb++){
                double *bP   = buffer + INDEX(0,jb,iDim);
                double *bN   = buffer + INDEX(0,jb+1,iDim);
                double *bS   = buffer + INDEX(0,jb-1,iDim);
                double *s    = sOut   + INDEX(0,j,iDim);
                double *sP   = sIn    + INDEX(0,j,iDim);
                #pragma ivdep
                for(int i=2; i<iDim-2; i++){
                    s[i] =  sP[i] - D*(-4.*bP[i] + bP[i-1] + bP[i+1] + bN[i] + bS[i]);
                }
            }
        }
    }
}

int main(int argc, char **argv){
    int iDim, jDim, jBlockDim, numTimeSteps;
    int i, j, run;
    // parameterized diffusion coefficient
    double D = 0.025;
    int printSolutions = 0;
    double totalTime;
    double *tmp;
    int numThreads = omp_get_max_threads();

    // read command line arguments
    if( !scanArgs(argc, argv, &iDim, &jDim, &numTimeSteps, &jBlockDim) )
        return -1;

    // adjust for boundary conditions:
    // fourth-order operator requires a boundary of width 2
    // on each edge
    iDim += 4;
    jDim += 4;
    printf( "dispersion benchmark with %d OpenMP threads\n", numThreads);
    printf( "grid with dimensions %dx%d, %d time steps, block height %d\n", iDim, jDim, numTimeSteps, jBlockDim);

    ////////////////////////////
    // allocate memory
    ////////////////////////////
    double *sIn    = (double*)malloc(iDim*jDim*sizeof(double));
    assert(sIn!=NULL);
    double *sOut   = (double*)malloc(iDim*jDim*sizeof(double));
    assert(sOut!=NULL);
    double *buffer = (double*)malloc(iDim*jDim*sizeof(double));
    assert(buffer!=NULL);

    // allocate thread-private buffers
    // used by method three
    double **buffers = (double**)malloc(numThreads*sizeof(double*));
    assert(buffers!=NULL);

    // initialise each thread-private buffers
    for(int tid=0; tid<numThreads; tid++)
    {
        buffers[tid] = (double*)malloc(iDim*(jBlockDim+2)*sizeof(double));
        assert(buffers[tid]!=NULL);
        double *mybuffer = buffers[tid];
        mybuffer = buffers[tid];
        for(int i=0; i<iDim*(jBlockDim+2); i++)
            mybuffer[i] = 0.;
    }

    ///////////////////////////////////////////
    // time loop for version 1
    ////////////////////////////////////////

    // initialise data fields
    setInitialConditions(sIn,  iDim, jDim);
    setInitialConditions(sOut, iDim, jDim);

    dispersionV1(sIn, sOut, buffer, D, iDim, jDim);
    totalTime = -omp_get_wtime();
    for(run=0; run<numTimeSteps; run++){
        // perform a single timestep
        dispersionV1(sIn, sOut, buffer, D, iDim, jDim);
        // swap the timesteps for the next iteration
        tmp  = sIn;
        sIn  = sOut;
        sOut = tmp;
    }
    totalTime += omp_get_wtime();
    printf( "\tversion 1 took %f seconds\n", totalTime);

    // print solution to screen if printSolutions is nonzero
    if(printSolutions)
        printSolution(sIn, iDim, jDim);

    ///////////////////////////////////////////
    // time loop for version 2
    ///////////////////////////////////////////

    // initialise data fields
    setInitialConditions(sIn,  iDim, jDim);
    setInitialConditions(sOut, iDim, jDim);

    dispersionV2(sIn, sOut, buffer, D, iDim, jDim, jBlockDim);
    totalTime = -omp_get_wtime();
    for(run=0; run<numTimeSteps; run++){
        // perform a single timestep
        dispersionV2(sIn, sOut, buffer, D, iDim, jDim, jBlockDim);
        if(printSolutions && run==numTimeSteps-1)
            printSolution(sOut, iDim, jDim);
        // swap the timesteps for the next iteration
        tmp  = sIn;
        sIn  = sOut;
        sOut = tmp;
    }
    totalTime += omp_get_wtime();
    printf( "\tversion 2 took %f seconds\n", totalTime);

    // print solution to screen if printSolutions is nonzero
    if(printSolutions)
        printSolution(sIn, iDim, jDim);

    ///////////////////////////////////////////
    // time loop for version 3
    ///////////////////////////////////////////

    // initialise data fields
    setInitialConditions(sIn,  iDim, jDim);
    setInitialConditions(sOut, iDim, jDim);

    dispersionV3(sIn, sOut, buffers, D, iDim, jDim, jBlockDim);
    totalTime = -omp_get_wtime();
    for(run=0; run<numTimeSteps; run++){
        // perform a single timestep
        dispersionV3(sIn, sOut, buffers, D, iDim, jDim, jBlockDim);
        // swap the timesteps for the next iteration
        tmp  = sIn;
        sIn  = sOut;
        sOut = tmp;
    }
    totalTime += omp_get_wtime();
    printf( "\tversion 3 took %f seconds\n", totalTime);

    // print solution to screen if printSolutions is nonzero
    if(printSolutions)
        printSolution(sIn, iDim, jDim);

    // free memory
    free(sIn);
    free(sOut);
    free(buffer);

    printf("============================================================\n");
}

int scanArgs(int argc, char** argv, int *nx, int *ny, int *nt, int *blocky){
    if(argc<5){
        printf("ERROR : not enough arguments\n\tusage $./laplace ni nj nt bDimj\n");
        return 0;
    }

    sscanf(argv[1], "%d", nx);
    sscanf(argv[2], "%d", ny);
    sscanf(argv[3], "%d", nt);
    sscanf(argv[4], "%d", blocky);
    if(*nx<1 || *nx>1024*1024){
        printf("ERROR : invalid x dimension : %d\n", *nx);
        return 0;
    }
    if(*ny<1 || *ny>1024*1024){
        printf("ERROR : invalid y dimension : %d\n", *ny);
        return 0;
    }
    if(*nt<1 || *nt>100000){
        printf("ERROR : invalid number of time steps : %d\n", *nt);
        return 0;
    }
    if(*blocky<1 || *ny%(*blocky) || *blocky>*ny){
        printf("ERROR : invalid block y dimension : %d\n", *blocky);
        return 0;
    }

    return 1;
}

