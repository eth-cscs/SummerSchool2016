for dim in 128 256 512 1024 2048 4096
do
    echo "=== $dim x $dim ==="
    printf "%8s%15s%15s%15s\n" threads "naiive" "global_block" "private_block"
    for threads in 1 2 4 8
    do
        OMP_NUM_THREADS=$threads aprun -d 8 ./a.out $dim $dim 50 16 > output
        tone=`grep "version 1" output | awk '{print $4}'`
        ttwo=`grep "version 2" output | awk '{print $4}'`
        tthree=`grep "version 3" output | awk '{print $4}'`
        printf "%8d%15.5f%15.5f%15.5f\n" $threads $tone $ttwo $tthree
    done
done
