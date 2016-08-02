#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define INDEX(I, J, IDIM) ((I)+(J)*(IDIM))

int scanArgs(int argc, char** argv, int *nx, int *ny, int *nt);
void printSolution(double *s, int iDim, int jDim);
void setInitialConditions(double *s, int iDim, int jDim);

//////////////////////////////////////////////////////////
// implementation of the laplace stencil loop
//
// The points in the stencil are labelled as points on the
// compas, with P in the centre:
//              N
//              |
//          W - P - E
//              |
//              S
//
// so the stencil is
//      sOut[P] = sP + D*(-4*sP + sN + sS + sE + sW)
//////////////////////////////////////////////////////////
// TODO : add openmp directives to parallelize
void laplace_base(double *sIn, double *sOut, double D, int iDim, int jDim){
#pragma omp parallel for
    for(int j=1; j<jDim-1; j++){
        double *sP   = sIn  + INDEX(0,j,iDim);
        double *sN   = sIn  + INDEX(0,j+1,iDim);
        double *sS   = sIn  + INDEX(0,j-1,iDim);
        double *s    = sOut + INDEX(0,j,iDim);
        for(int i=1; i<iDim-1; i++){
            s[i] =  sP[i] + D*(-4.*sP[i] + sP[i-1] + sP[i+1] + sN[i] + sS[i]);
        }
    }
}

// TODO : add openmp directives to parallelize
// TODO : make it vectorize
void laplace_vectorize(double *sIn, double *sOut, double D, int iDim, int jDim){
#pragma omp parallel for
    for(int j=1; j<jDim-1; j++){
        double *sP   = sIn  + INDEX(0,j,iDim);
        double *sN   = sIn  + INDEX(0,j+1,iDim);
        double *sS   = sIn  + INDEX(0,j-1,iDim);
        double *s    = sOut + INDEX(0,j,iDim);
        #pragma ivdep
        for(int i=1; i<iDim-1; i++){
            s[i] =  sP[i] + D*(-4.*sP[i] + sP[i-1] + sP[i+1] + sN[i] + sS[i]);
        }
    }
}

int main(int argc, char **argv){
    int iDim, jDim, numTimeSteps;
    int i, j, run;
    double D = 0.25;        // parameterized diffusion coefficient
    double totalTime;
    double *tmp;

    if( !scanArgs(argc, argv, &iDim, &jDim, &numTimeSteps) )
        return -1;
    // adjust for boundary conditions
    iDim += 2;
    jDim += 2;
    printf( "laplace benchmark with %d OpenMP threads\n", omp_get_max_threads());
    printf( "grid with dimensions %dx%d, %d time steps\n", iDim, jDim, numTimeSteps);

    ////////////////////////////
    // allocate memory
    ////////////////////////////
    double *sIn  = (double*)malloc(iDim*jDim*sizeof(double));
    double *sOut = (double*)malloc(iDim*jDim*sizeof(double));

    // the top and bottom boundaries are set to 1
    setInitialConditions(sIn,  iDim, jDim);
    setInitialConditions(sOut, iDim, jDim);

    laplace_base(sIn, sOut, D, iDim, jDim);
    totalTime = -omp_get_wtime();
    for(run=0; run<numTimeSteps; run++){
        // perform a single timestep
        laplace_base(sIn, sOut, D, iDim, jDim);
        // swap the timesteps for the next iteration
        tmp  = sIn;
        sIn  = sOut;
        sOut = tmp;
    }
    totalTime += omp_get_wtime();
    printf( "laplace_base       took %f seconds\n", totalTime);
    //printSolution(sOut, iDim, jDim);

    laplace_vectorize(sIn, sOut, D, iDim, jDim);
    totalTime = -omp_get_wtime();
    for(run=0; run<numTimeSteps; run++){
        // perform a single timestep
        laplace_vectorize(sIn, sOut, D, iDim, jDim);

        // swap the timesteps for the next iteration
        tmp  = sIn;
        sIn  = sOut;
        sOut = tmp;
    }
    totalTime += omp_get_wtime();
    printf( "laplace_vectorized took %f seconds\n", totalTime);

    // free memory
    free(sIn);
    free(sOut);
}

int scanArgs(int argc, char** argv, int *nx, int *ny, int *nt){
    if(argc<4){
        printf("ERROR : not enough arguments\n\tusage $./laplace nx ny nt\n");
        return 0;
    }

    sscanf(argv[1], "%d", nx);
    sscanf(argv[2], "%d", ny);
    sscanf(argv[3], "%d", nt);
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

    return 1;
}

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

    // the top and bottom boundaries are set to 1
    for(i=0; i<iDim; i++){
        s[i] = 1.;
        s[i+(jDim-1)*iDim]  = 1.;
    }
    for(j=1; j<jDim-1; j++){
        int k=j*iDim+1;
        // set left boundary to 1
        s[k-1] = 1.;
        // set interior to 0
        for(i=1; i<iDim-1; i++, k++){
            s[k] = 0.;
        }
        // set right boundary to 1
        s[k] = 1.;
    }
}

