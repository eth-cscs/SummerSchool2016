for n in `seq 12 2 24`
do
    echo === length 2^$n
    OMP_NUM_THREADS=8 aprun -cc none ./axpy.omp $n | grep ^::
    echo
done
