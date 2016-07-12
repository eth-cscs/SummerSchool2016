for dim in 64 128 256
do
    echo ======= ${dim}x${dim}
    for t in 1 2 4 8
    do
        printf "%3d threads : "  $t
        OMP_NUM_THREADS=$t srun -n1 -c8 --hint=nomultithread ./main $dim $dim 100 0.01 | grep took;
    done
done
