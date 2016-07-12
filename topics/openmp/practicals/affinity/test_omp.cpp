#include <iomanip>
#include <iostream>
#include <sstream>

#include <omp.h>

#include "affinity.h"

int main(void) {

    auto num_threads = omp_get_max_threads();
    std::vector<std::string> strings(num_threads);

    #pragma omp parallel
    {
        auto cores = get_affinity();
        std::stringstream s;
        for(auto core: cores) {
            s << std::setw(3) << core << " ";
        }
        auto thread_id = omp_get_thread_num();
        strings[thread_id] = s.str();
    }

    for(auto i=0; i<num_threads; ++i) {
        std::cout << "thread " << std::setw(3) << i
                  << " : " << strings[i]
                  << std::endl;
    }
}
