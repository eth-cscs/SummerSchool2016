#include "data.h"

#include <stdio.h>

namespace data {
// fields that hold the solution
Field x_new;
Field x_old;

// fields that hold the boundary points
Field bndN;
Field bndE;
Field bndS;
Field bndW;

Discretization options;

std::ostream &operator<<(std::ostream &out, Field &field)
{
    auto print_elem = [](std::ostream &out,
                         const double &val, int count, int pos, int len) {
        out << val;
        if (count > 1) {
            out << "{" << count << "}";
        }

        out << " ";
    };

    field.update_host();

    auto len = field.length();
    out << "[ ";
    if (len) {
        auto count = 1;
        auto last_elem = field.ptr_[0];
        for (auto i = 1; i < len; ++i) {
            if (field.ptr_[i] != last_elem) {
                print_elem(out, field.ptr_[i], count, i, len);
                count = 1;
                last_elem = field.ptr_[i];
            } else {
                ++count;
            }
        }

        if (count > 1) {
            print_elem(out, field.ptr_[len-1], count, len-1, len);
        }
    }

    out << "]";
    return out;
}

} //end of namespace data
