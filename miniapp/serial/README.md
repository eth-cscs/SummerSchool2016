To compile and run the serial version of the miniapp

```bash
# choose your preferred version (C++ in this example)
cd cxx

# choose your compiler
# if you skip this step the default Cray compiler will be used
module swap PrgEnv-cray PrgEnv-gnu

# make the code
make

# run once to see the available options
srun ./main

# now run on a 128x128 grid to t=0.01 via 100 time steps
srun ./main 128 128 100 0.01
```

Benchmark results on Piz Daint `srun main 128 128 100 0.01`

Measured in CG iterations/second.

```
           ---------------------------------
           | cray     gnu    intel    pgi  |
--------------------------------------------
| C++      |  7508    8020    8013    4101 |
| Fortran  | 10678    7658   10374    9949 |
--------------------------------------------
```
