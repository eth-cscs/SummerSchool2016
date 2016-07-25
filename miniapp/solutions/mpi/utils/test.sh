#!/bin/bash

echo "Only MPI"
for dim in 256
do
    echo ======= ${dim}x${dim}
    for p in 1 2 4 8 16 32
    do
       n=$p
       if (( $p > 4 )); then
           n=4
       fi
       printf "%3d nodes, %3d ranks [ %3d ]: "  $n $p $p
       srun -n$p -N$n --hint=nomultithread ./main $dim $dim 100 0.01 > output_miniapp
       time=$(grep took output_miniapp | awk '{print $3 " " $4};')
       iter=$(grep rate output_miniapp | awk '{print $8 " " $9};')
       echo $time $iter
       #srun -n$p -N$n --hint=nomultithread hostname
    done
done

echo "MPI+OpenMP"
for dim in 256
do
    echo ======= ${dim}x${dim}
    for p in 1 2 4
    do
        for t in 1 2 4 8
        do
            tp=$(( $t*$p ))
            printf "%3d nodes, %3d ranks,  %3d omp threads [ %3d ] : "  $p $p $t $tp
            OMP_NUM_THREADS=$t srun -n$p -N$p -c8 --hint=nomultithread ./main $dim $dim 100 0.01 > output_miniapp
            time=$(grep took output_miniapp | awk '{print $3 " " $4};')
            iter=$(grep rate output_miniapp | awk '{print $8 " " $9};')
            echo $time $iter
        done
    done
done

rm output_miniapp
