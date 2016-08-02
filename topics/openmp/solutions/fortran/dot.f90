program dot

use omp_lib

implicit none

integer      :: N = 100000000
integer      :: i, num_threads
real(kind=8), dimension(:), allocatable  :: a, b
real(kind=8) :: time, sum, sum_local, expected

allocate(a(N), b(N))

num_threads = omp_get_max_threads()
print *, 'dot of vectors with length ', N, ' with ', num_threads, ' threads'

do i=1,N
    a(i) = 1.0d0/2.0d0
    b(i) = i
enddo

time = -omp_get_wtime();

sum = 0.0d0
!$omp parallel private(sum_local)
    sum_local = 0.0d0
    !$omp do
    do i=1,n
        sum_local = sum_local + a(i) * b(i)
    end do
    !$omp end do

    !$omp critical
    sum = sum + sum_local
    !$omp end critical
!$omp end parallel

time = time + omp_get_wtime()

expected = (N+1.0d0)*N/4.0d0;
print *, 'relative error ', abs(expected-sum)/expected
print *, 'took ', time, ' seconds'

deallocate(a, b)

end ! program dot

