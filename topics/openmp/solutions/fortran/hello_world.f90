program hello_world

use omp_lib

integer tid

print *, '=== serial section ==='

print *, 'hello world from thread ', omp_get_thread_num(), ' of ', omp_get_num_threads()

print *, '=== parallel section ==='

!$omp parallel private(tid)

    ! get the number of this thread
    tid = omp_get_thread_num();

    ! write a personalized message from this thread
    !$omp critical
    print *, 'hello world from thread ', tid, ' of ', omp_get_num_threads()
    !$omp end critical

!$omp end parallel

end

