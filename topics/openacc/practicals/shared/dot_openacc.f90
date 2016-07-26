subroutine dotprod_gpu(n, x, y, res)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(in) :: y(n)
  real(kind(0d0)), intent(out) :: res

  res = 0
  ! TODO: implement dot prod with OpenACC

end subroutine dotprod_gpu

subroutine dotprod_host(n, x, y, res)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(in) :: y(n)
  real(kind(0d0)), intent(out) :: res

  res = 0
  !$omp parallel do reduction(+:res)
  do i = 1,n
     res = res + x(i)*y(i)
  enddo

end subroutine dotprod_host

program main
  use util
  implicit none

  integer n, err
  real(kind(0d0)), dimension(:), allocatable :: x, y
  real(kind(0d0)) :: result, expected

  n = read_arg(1, 4)

  write(*, '(a i0)') 'dot product of length n = ', n
  allocate(x(n), y(n), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  call random_seed()

  x(:) = 2.0d0
  call random_number(y)
  y(:) = 10*y(:)

  call dotprod_gpu(n, x, y, result)
  call dotprod_host(n, x, y, expected)

  if (abs(expected - result) > 2*n*1d-14) then
     write(*, '(a f0.6 a f0.6)') '============ FAILED: got ', result, &
          ' expected: ', expected
  else
     write(*, '(a)') '============ SUCCESS'
  endif

end program main
