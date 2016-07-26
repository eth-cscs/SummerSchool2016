subroutine axpy(n, alpha, x, y)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: alpha
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(inout) :: y(n)

  integer i

  !$omp parallel do
  do i = 1,n
     y(i) = y(i) + alpha*x(i)
  enddo

end subroutine axpy

subroutine axpy_gpu(n, alpha, x, y)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: alpha
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(inout) :: y(n)

  ! TODO: Implement the axpy kernel using OpenACC

end subroutine axpy_gpu

program main
  use util
  implicit none

  integer pow, n, err, i
  real(kind(0d0)), dimension(:), allocatable :: x, x_, y, y_
  real(kind(0d0)) :: axpy_start, time_axpy

  pow = read_arg(1, 16)
  n = 2**pow
  print *, 'memcopy and daxpy test of size', n
  allocate(x(n), y(n), x_(2**24), y_(2**24), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  x(:)  = 1.5d0
  y(:)  = 3.0d0
  x_(:) = 1.5d0
  y_(:) = 3.0d0
  call axpy(2**24, 2d0, x_, y_)

  axpy_start = get_time()
  call axpy_gpu(n, 2d0, x, y)

  ! TODO: wait for stream to finish
  time_axpy = get_time() - axpy_start

  print *, '-------'
  print *, 'timings'
  print *, '-------'
  print *, 'axpy    : ', time_axpy, 's'

  err=0
  !$omp parallel do reduction(+:err)
  do i = 1,n
     if (abs(6d0 - y(i)) > 1d-15) then
        err = err + 1
     endif
  enddo

  if (err > 0) then
     print *, '============ FAILED with ', err, ' errors'
  else
     print *, '============ PASSED'
  endif

  deallocate(x, y)

end program main
