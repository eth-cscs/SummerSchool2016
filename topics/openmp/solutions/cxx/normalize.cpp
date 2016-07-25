#include <iostream>
#include <cstdlib>
#include <cmath>

#include <omp.h>

// function to compute the 2-norm of a vector v of length n
double norm(double *v, int n){
    double norm = 0.;

    for(int i=0; i<n; i++)
        norm += v[i]*v[i];

    return sqrt(norm);
}

// initialise v to values between -10 and 10
void initialize(double *v, int n){
    for(int i=0; i<n; i++)
        v[i] = cos(double(i)) * 10.;
}


void normalize_vector(double *v, int n){
    double norm = 0.;

    // compute the norm of v
    for(int i=0; i<n; i++)
        norm += v[i]*v[i];
    norm = sqrt(norm);

    // normalize v
    for(int i=0; i<n; i++)
        v[i] /= norm;
}

void normalize_vector_omp_1(double *v, int n) {
    double norm = 0.;

    #pragma omp parallel
    {
        double norm_local = 0.;
        #pragma omp for
        for(int i=0; i<n; i++)
            norm_local += v[i]*v[i];

        #pragma omp critical
        norm += norm_local;

        #pragma omp barrier

        #pragma omp single
        norm = sqrt(norm);

        #pragma omp for
        for(int i=0; i<n; i++)
            v[i] /= norm;
    }
}

// use reduction clause and one parallel block
void normalize_vector_omp_2(double *v, int n) {
    double norm = 0.;

    #pragma omp parallel
    {
        #pragma omp for reduction(+:norm)
        for(int i=0; i<n; i++)
            norm += v[i]*v[i];

        #pragma omp single
        norm = sqrt(norm);

        #pragma omp for
        for(int i=0; i<n; i++)
            v[i] /= norm;
    }
}

// reduction clause with separate parallel regions
void normalize_vector_omp_3(double *v, int n) {
    double norm = 0.;

    #pragma omp parallel for reduction(+:norm)
    for(int i=0; i<n; i++)
        norm += v[i]*v[i];

    norm = sqrt(norm);

    #pragma omp parallel for
    for(int i=0; i<n; i++)
        v[i] /= norm;
}

int main( void ){
    const int N = 40000000;
    double *v = (double*)malloc(N*sizeof(double));
    bool validated = false;

    initialize(v, N);
    double time_serial = -omp_get_wtime();
    normalize_vector(v, N);
    time_serial += omp_get_wtime();

    // chck the answer
    std::cout << "serial error   : " << fabs(norm(v,N) - 1.) << std::endl;

    int max_threads = omp_get_max_threads();
    initialize(v, N);
    double time_parallel = -omp_get_wtime();
    normalize_vector_omp_1(v, N);
    //normalize_vector_omp_2(v, N);
    time_parallel += omp_get_wtime();

    // chck the answer
    std::cout << "parallel error : " << fabs(norm(v,N) - 1.) << std::endl;

    std::cout << max_threads     << " threads" << std::endl;
    std::cout << "serial     : " << time_serial << " seconds\t"
              << "parallel   : " << time_parallel <<  " seconds" << std::endl;
    std::cout << "speedup    : " << time_serial/time_parallel << std::endl;
    std::cout << "efficiency : " << (time_serial/time_parallel)/double(max_threads) << std::endl;

    free(v);
    return 0;
}


