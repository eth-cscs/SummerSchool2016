#include <iostream>

#include <cstdlib>

#include <omp.h>

double histogram_serial(const int *values,
                        int *bins,
                        const int nbins,
                        const int n)
{
    double time = -omp_get_wtime();

    // initialize bins to zero
    for(int i=0; i<nbins; ++i) {
        bins[i] = 0;
    }

    // count values
    for(int i=0; i<n; ++i) {
        bins[values[i]]++;
    }

    time += omp_get_wtime();
    return time;
}

// TODO : find the performance bug in this function
// hint : the bug only affects performance for num_threads>1
double histogram_omp(const int *values,
                     int *bins,
                     const int nbins,
                     const int n)
{
    double time = -omp_get_wtime();

    const int num_threads= omp_get_max_threads();
    int par_bins[num_threads][nbins];

    #pragma omp parallel
    {
        int this_thread = omp_get_thread_num();
        int* my_bins=par_bins[this_thread];
        for(int i=0; i<nbins; ++i) {
            my_bins[i] = 0;
        }

        #pragma omp for
        for(int i=0; i<n; ++i) {
            my_bins[values[i]]++;
        }
    }

    // do reduction at end
    for(int j=0; j<nbins; ++j) {
        bins[j] = 0;
        for(int i=0; i<num_threads; ++i) {
            bins[j] += par_bins[i][j];
        }
    }

    time += omp_get_wtime();
    return time;
}

int main(void) {
    const int n = 1<<29;
    const int nbins = 2;

    // initialize counts in the bins to zero
    int bins[nbins];

    // allocate memory for values and initialize to
    // random values in range [0, nbins-1]
    int *values = new int[n];
    for(int i=0; i<n; ++i) {
        values[i] = rand()%nbins;
    }

    // do serial run
    double serial_time = histogram_serial(values, bins, nbins, n);

    std::cout << "serial run results\n------------------\n";
    for(int i=0; i<nbins; ++i) {
        std::cout << i << "\t" << bins[i] << std::endl;
    }

    // do parallel run
    double parallel_time = histogram_omp(values, bins, nbins, n);

    std::cout << "\nparallel run results\n------------------\n";
    for(int i=0; i<nbins; ++i) {
        std::cout << i << "\t" << bins[i] << std::endl;
    }

    std::cout << std::endl;
    std::cout << "serial   : " << serial_time << " seconds" << std::endl;
    std::cout << "parallel : " << parallel_time << " seconds" << std::endl;
}

