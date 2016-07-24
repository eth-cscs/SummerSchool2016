This is a version of the miniapp which uses PETSc. It is different in spirit from
the other versions of the miniapp, in that it is not intended to be an 
essentially-identical piece of code with to-be-ported kernels, but is rather
an example of writing an equivalent code using a higher-level library.

For more information, see the notes in the .c files here

Quick start on Piz Daint (updated 2016.07.24):

1. Log in, via ssh with X tunneling, to daint
    ssh -X courseNN@ela.cscs.ch   # replace "courseNN" with your CSCS username
    ssh -X daint

2. Load modules
    module list   # confirm that you have the PrgEnv-cray module loaded
    module load cray-petsc
    
3. Clone this repository and navigate to this directory (if needbe).
    cd $SCRATCH   # or choose another location
    git clone https://github.com/eth-cscs/SummerSchool2016
    cd SummerSchool2016/miniapp/petsc

3. Build the executable
    make

4. Test in an interactive session 
    salloc
    module load cray-petsc
    make test

   You should see
    Running Test 1
    Success
    Running Test 2
    Success

5. Run your own experiments in the interactive session
    srun -n 4 ./main -nx 99 -ny 88 -ts_monitor -snes_monitor -ksp_monitor 
    srun -n 1 ./main -nx 16 -ny 16 -ts_monitor -snes_monitor -ksp_monitor -assemble 1 -pc_type gamg -dump 1

   You can also modify the included job.daint and submit with sbatch

   To view the .bov file that is generated (only for single-processor runs with the -dump option), we borrow the procedure from the miniapp
    module load python/2.7.6
    python plotting.py
    display output.png  # make sure that you used ssh -X to log in

   Note: you will get two harmless UnicodeWarning messages, which can be ignored
