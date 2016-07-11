dim=256
steps=200

make clean main > cmp
cp main main.cray
aprun main.cray $dim $dim $steps 0.01 &> tmp
echo ============= Cray ===============
grep second tmp
grep iteration tmp

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/4.9.2
make clean main > cmp
cp main main.gnu
aprun main.gnu $dim $dim $steps 0.01 &> tmp
echo ============= GNU ===============
grep second tmp

module swap PrgEnv-gnu PrgEnv-intel
make clean main > cmp
cp main main.intel
aprun main.intel $dim $dim $steps 0.01 &> tmp
echo ============= Intel ===============
grep second tmp

module swap PrgEnv-intel PrgEnv-pgi
make clean main > cmp
cp main main.pgi
aprun main.pgi $dim $dim $steps 0.01 &> tmp
echo ============= PGI ===============
grep second tmp
