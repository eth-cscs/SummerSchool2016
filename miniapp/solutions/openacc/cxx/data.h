#ifndef DATA_H
#define DATA_H

#include <iostream>

#define DEBUG_PRINT_VAR(out, v) out << "[DEBUG]: " << #v << " = " << v

namespace data
{
// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int nt;       // number of time steps
    int N;        // total number of grid points
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

extern Discretization options;

// thin wrapper around a pointer that can be accessed as either a 2D or 1D array
// Field has dimension xdim * ydim in 2D, or length=xdim*ydim in 1D
class Field {
    public:
    // default constructor
    Field()
    :   ptr_(0), xdim_(0), ydim_(0)
    {};

    // constructor
    Field(int xdim, int ydim)
    :   xdim_(xdim), ydim_(ydim)
    {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif

        ptr_ = new double[xdim*ydim];
        #pragma acc enter data copyin(this)
        #pragma acc enter data create(ptr_[0:xdim*ydim])
    };

    // destructor
    ~Field() {
        free();
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        free();
        ptr_ = new double[xdim*ydim];
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        xdim_ = xdim;
        ydim_ = ydim;

        #pragma acc enter data copyin(this)
        #pragma acc enter data create(ptr_[0:xdim*ydim])
    }

    void update_host()
    {
        auto ptr = ptr_;
        #pragma acc update host(ptr[0:xdim_*ydim_])
    }

    void update_device()
    {
        auto ptr = ptr_;
        #pragma acc update device(ptr[0:xdim_*ydim_])
    }

    double* data()
    {
        return ptr_;
    }

    const double* data() const {
        return ptr_;
    }

    // access via (i,j) pair
    #pragma acc routine seq
    inline double& operator() (int i, int j) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    #pragma acc routine seq
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    // access as a 1D field
    #pragma acc routine seq
    inline double& operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    #pragma acc routine seq
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    int xdim()   const { return xdim_; }
    int ydim()   const { return ydim_; }
    int length() const { return xdim_*ydim_; }

    friend std::ostream &operator<<(std::ostream &out, Field &field);

    private:

    void free() {
         if (ptr_) {
             #pragma acc exit data delete(ptr_[0:xdim_*ydim_], this)
             delete[] ptr_;
        }

        ptr_ = 0;
    }

    double* ptr_;
    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new, x_old; // 2d
extern Field bndN, bndE, bndS, bndW; // 1d

}

#endif // DATA_H
