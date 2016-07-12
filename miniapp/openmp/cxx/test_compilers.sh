dim=256
steps=200

make clean main > cmp
OMP_NUM_THREADS=8 aprun  -cc none main $dim $dim $steps 0.01 &> tmp
echo ============= Cray ===============
grep second tmp
grep iteration tmp

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/4.9.2
make clean main > cmp
OMP_NUM_THREADS=8 aprun  -cc none main $dim $dim $steps 0.01 &> tmp
echo ============= GNU ===============
grep second tmp

module swap PrgEnv-gnu PrgEnv-intel
make clean main > cmp
OMP_NUM_THREADS=8 aprun  -cc none main $dim $dim $steps 0.01 &> tmp
echo ============= Intel ===============
grep second tmp

module swap PrgEnv-intel PrgEnv-pgi
make clean main > cmp
OMP_NUM_THREADS=8 aprun  -cc none main $dim $dim $steps 0.01 &> tmp
echo ============= PGI ===============
grep second tmp

rm -f tmp
rm -f cmp
