PROGRAM coordinated

implicit none
include "mpif.h"

integer, parameter :: nsize = 1000000
real, allocatable, dimension(:) :: data, fulldata
integer i, rank, size, ierror, tag, status(MPI_STATUS_SIZE)
real*8 times(10)

!switch on MPI   
call MPI_INIT(ierror)

!get MPI rank and number of processes
call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

!allocate main array
allocate(data(nsize))

!initialize array
do i=1,nsize
   data(i) = real(rank)
enddo

!write data (one file only 

times=0.0
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(1) = MPI_Wtime()
open(unit=1000, access="direct", recl=nsize*4)
write(1000, rec=rank+1)data
close(1000)
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(2) = MPI_Wtime()
if(rank == 0)write(*,*)"Time for I/O (sec) = ", times(2) - times(1)

!switch off MPI
call MPI_FINALIZE(ierror)


END PROGRAM coordinated
