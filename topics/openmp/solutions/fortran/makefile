all: dot.exe normalize.exe hello_world.exe sum_threads.exe pi.exe

args=

pi.exe : pi.f90
	ftn ${args} pi.f90 -fopenmp -o pi.exe

normalize.exe : normalize.f90
	ftn ${args} normalize.f90 -fopenmp -o normalize.exe

hello_world.exe : hello_world.f90
	ftn ${args} hello_world.f90 -fopenmp -o hello_world.exe

dot.exe : dot.f90
	ftn ${args} dot.f90 -fopenmp -o dot.exe

sum_threads.exe : sum_threads.f90
	ftn ${args} sum_threads.f90 -fopenmp -o sum_threads.exe

clean :
	rm -f *.exe

