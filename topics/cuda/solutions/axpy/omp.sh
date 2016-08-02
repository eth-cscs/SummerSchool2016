for n in `seq 12 2 24`
do
    echo === length 2^$n
    OMP_NUM_THREADS=8 srun -c8 -n1 --hint=nomultithread ./axpy.omp $n | grep ^::
    echo
done
