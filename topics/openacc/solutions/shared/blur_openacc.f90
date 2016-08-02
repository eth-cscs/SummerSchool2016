real(kind(0d0)) function blur(pos, u, n)
  !$acc routine seq
  integer, intent(in) :: pos, n
  real(kind(0d0)), intent(in) :: u(n)

  blur = 0.25*(u(pos-1) + 2.0*u(pos) + u(pos+1))
end function blur

subroutine blur_twice_host(nsteps, n, in, out)
  use util
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur
  !$acc routine(blur) seq

  allocate(buffer(n))

  do istep = 1,nsteps
     !$omp parallel do
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     !$omp parallel do
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo
     call swap(in, out)
  enddo

  deallocate(buffer)
end subroutine blur_twice_host

subroutine blur_twice_gpu_naive(nsteps, n, in, out)
  use util
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur
  !$acc routine(blur) seq

  allocate(buffer(n))

  do istep = 1,nsteps
     !$acc parallel loop copyin(in) copyout(buffer)
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     !$acc parallel loop copyin(buffer) copy(out)
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo

     call swap(in, out)
  enddo

  deallocate(buffer)
end subroutine blur_twice_gpu_naive

subroutine blur_twice_gpu_nocopies(nsteps, n, in, out)
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur
  !$acc routine(blur) seq

  allocate(buffer(n))

  !$acc data copyin(in) copy(out) create(buffer)
  do istep = 1,nsteps
     !$acc parallel
     !$acc loop independent
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     !$acc loop independent
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo

     !$acc loop independent
     do i = 1,n
        in(i) = out(i)
     enddo
     !$acc end parallel
  enddo
  !$acc end data

  deallocate(buffer)
end subroutine blur_twice_gpu_nocopies

program main
  use util
  implicit none

  integer pow, n, nsteps, err, i
  real(kind(0d0)), dimension(:), allocatable :: x0, x1
  real(kind(0d0)) :: blur_start, time_blur
  pow    = read_arg(1, 20)
  nsteps = read_arg(2, 100)
  n      = 2**pow + 4

  write(*, '(a i0 a f0.6 a)') 'dispersion 1D test of length n = ', n, ' : ', 8.*n/1024**2, 'MB'

  allocate(x0(n), x1(n), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  x0(1)   = 1.0
  x0(2)   = 1.0
  x0(n-1) = 1.0
  x0(n)   = 1.0

  x1(1)   = x0(1)
  x1(2)   = x0(2)
  x1(n-1) = x0(n-1)
  x1(n)   = x0(n)

  blur_start = get_time()
  call blur_twice_gpu_nocopies(nsteps, n, x0, x1)
  time_blur = get_time() - blur_start

  write(*, '(a f0.6 a f0.8 a)') '==== that took ', time_blur, &
       ' seconds (', time_blur/nsteps, 's/step )'
  do i = 1, min(20,n)
     write(*, '(f0.6 a)', advance='no') x1(i), ' '
  enddo
  write(*,*)

end program main
